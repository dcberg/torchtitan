# Running TorchTitan on Kubernetes with Kueue

TorchTitan does not include a Kubernetes-native launcher. The recommended batch pattern is:

1. Build and push a container image with TorchTitan and the matching PyTorch stack.
2. Use Kueue for queue admission.
3. Use JobSet as the workload controller for multi-node `torchrun`.

The example manifests are in:

- `k8s/kueue-localqueue.yaml`
- `k8s/torchtitan-kueue-jobset.yaml`

## Why JobSet instead of StatefulSet

`StatefulSet` can be admitted by Kueue, but it is a serving-oriented controller. For multi-node training jobs, `JobSet` is a better fit because it models a distributed batch workload directly and gives each worker a stable DNS name suitable for `torchrun` rendezvous.

## Prerequisites

- Kueue installed in the cluster.
- JobSet installed in the cluster.
- A `ClusterQueue` already exposed to your namespace.
- GPU nodes with working NCCL pod-to-pod connectivity.
- A container image that can run `torchrun -m torchtitan.train`.
- Shared or pre-staged storage for tokenizer assets, datasets, and checkpoints.

## Queue setup

The sample `LocalQueue` points to a placeholder cluster queue:

```yaml
spec:
  clusterQueue: gpu-batch
```

Replace `gpu-batch` with the queue your cluster admin provides.

## Launch model

The sample JobSet uses one child Job per node:

- `spec.replicatedJobs[0].replicas` = total node count
- `NPROC_PER_NODE` = GPUs per node
- `NNODES` = total node count

JobSet creates stable pod hostnames. The sample command derives the node rank from the hostname and uses rank 0 as the rendezvous endpoint:

```bash
NODE_RANK="$(echo "${HOSTNAME}" | awk -F- '{print $(NF-1)}')"
MASTER_ADDR="${JOBSET_NAME}-trainer-0-0.${JOBSET_NAME}"
```

That is enough for `torchrun`:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  -m torchtitan.train \
  --module "${MODULE}" \
  --config "${CONFIG}"
```

## Applying the manifests

Adjust these fields before submitting:

- `image`
- `replicas`
- `NNODES`
- `NPROC_PER_NODE`
- `MODULE`
- `CONFIG`
- PVC name in `claimName`
- CPU and memory requests

Then apply:

```bash
kubectl apply -f k8s/kueue-localqueue.yaml
kubectl apply -f k8s/torchtitan-kueue-jobset.yaml
```

## Operational notes

- Keep `replicas` and `NNODES` in sync.
- Keep `resources.limits["nvidia.com/gpu"]` and `NPROC_PER_NODE` in sync.
- Avoid downloading Hugging Face assets independently in every pod; stage them on the image or a shared volume.
- If you need retries, increase `failurePolicy.maxRestarts` and make sure checkpointing is configured.
- If you need topology-aware placement, add the relevant Kueue podset topology annotations to the pod template.
