# Running TorchTitan on Kubernetes with Kueue

TorchTitan does not include a Kubernetes-native launcher. The recommended batch pattern is:

1. Build and push a container image with TorchTitan and the matching PyTorch stack.
2. Use Kueue for queue admission.
3. Use MPI Operator as the workload controller for multi-node training.

The example manifests are in:

- `k8s/kueue-localqueue.yaml`
- `k8s/torchtitan-kueue-jobset.yaml`
- `k8s/torchtitan-kueue-mpijob.yaml`
- `k8s/torchtitan-oke-fss-pvc.yaml`

A PHX-specific example for the `gpu-a10-2` OKE node pool is documented in `docs/kubernetes-phx-a10.md` with manifests in `k8s/phx-a10/`.

That PHX directory now contains separate validated smoke-test manifests plus separate larger-GPU templates:

- `torchtitan-jobset.yaml`: torchrun-on-JobSet smoke test for the A10 pool
- `torchtitan-mpijob.yaml`: smoke-test MPIJob for the A10 pool
- `torchtitan-jobset-llama3-8b.yaml`: suspended JobSet template for a larger GPU pool
- `torchtitan-mpijob-llama3-8b.yaml`: suspended template for a larger GPU pool

Both controller styles are intentionally kept in the repo:

- `JobSet` is the simpler `torchrun` path when stable pod DNS is enough.
- `MPIJob` is the launcher-plus-workers path when you specifically want MPI Operator semantics.

## Prerequisites

- Kueue installed in the cluster.
- MPI Operator installed in the cluster.
- A `ClusterQueue` already exposed to your namespace.
- GPU nodes with working NCCL pod-to-pod connectivity.
- A container image that can run `mpirun` in the launcher and `sshd` in the workers.
- Shared or pre-staged storage for tokenizer assets, datasets, and checkpoints.
- An OKE File Storage Service CSI `StorageClass` for the shared PVC, such as a class backed by `fss.csi.oraclecloud.com`.

## Install Kueue, JobSet, and MPI Operator

If your cluster does not already have Kueue, JobSet, and MPI Operator, install them before applying the TorchTitan manifests:

```bash
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.16.1 \
  --namespace kueue-system \
  --create-namespace \
  --wait --timeout 300s

helm install jobset oci://registry.k8s.io/jobset/charts/jobset \
  --version=0.11.1 \
  --namespace jobset-system \
  --create-namespace \
  --wait --timeout 300s

helm repo add kubeflow https://kubeflow.github.io/mpi-operator
helm repo update
helm install mpi-operator kubeflow/mpi-operator \
  --namespace mpi-operator \
  --create-namespace \
  --wait --timeout 300s
```

## Queue setup

The sample `LocalQueue` points to a placeholder cluster queue:

```yaml
spec:
  clusterQueue: gpu-batch
```

Replace `gpu-batch` with the queue your cluster admin provides.

## Shared storage on OKE

The sample MPIJob mounts one shared claim across all worker pods. For multi-node training on OKE, that means the claim should be `ReadWriteMany` rather than a block-volume `ReadWriteOnce` claim.

The repository now includes an OKE-oriented PVC example:

- `k8s/torchtitan-oke-fss-pvc.yaml`

It expects an existing OKE File Storage Service CSI `StorageClass` and defaults to:

- `storageClassName: fss-dyn-storage`
- `accessModes: [ReadWriteMany]`
- `storage: 50Gi`

If your cluster uses a different FSS-backed storage class name, change `storageClassName` in the PVC manifest before applying it.

## Launch model

The sample JobSet uses one trainer pod per node:

- `replicatedJobs[0].replicas` = total node count
- `NNODES` = total node count
- `NPROC_PER_NODE` = GPUs per node

JobSet gives each trainer pod a stable hostname. The sample command derives `NODE_RANK` from that hostname and uses rank 0 as the rendezvous endpoint:

```bash
NODE_RANK="$(echo "${HOSTNAME}" | awk -F- '{print $(NF-1)}')"
MASTER_ADDR="${JOBSET_NAME}-trainer-0-0.${JOBSET_NAME}"

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

The sample MPIJob uses:

- one `Launcher` replica that runs `mpirun`
- `Worker.replicas` = total node count
- `slotsPerWorker` = GPUs per node

The launcher resolves `MASTER_ADDR` from the first line of `/etc/mpi/hostfile` and exports the Torch distributed env vars on each rank from the Open MPI runtime env:

```bash
MASTER_ADDR="$(awk 'NR==1 {print $1}' /etc/mpi/hostfile)"

mpirun ... bash -lc '
  export RANK="${OMPI_COMM_WORLD_RANK}"
  export WORLD_SIZE="${OMPI_COMM_WORLD_SIZE}"
  export LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"
  exec python -m torchtitan.train \
    --module "${MODULE}" \
    --config "${CONFIG}"
'
```

The launcher and workers also stage the MPI Operator SSH secret into writable directories under `/run`. That avoids two failure modes that showed up in-cluster:

- the secret mounted at `/root/.ssh` is not a good place to write an SSH client config
- worker `sshd` strict-mode checks can reject `authorized_keys` if you relocate it under `/tmp`

The sample manifest also passes `--hostfile /etc/mpi/hostfile` explicitly so Open MPI does not start local ranks inside the launcher pod.

## Applying the manifests

Adjust these fields before submitting:

- `image`
- `MODULE`
- `CONFIG`
- PVC name in `claimName`
- OKE FSS storage class name in `k8s/torchtitan-oke-fss-pvc.yaml`
- CPU and memory requests

For JobSet, also adjust:

- `metadata.name` and `JOBSET_NAME` together
- `replicatedJobs[0].replicas`
- `NNODES`
- `NPROC_PER_NODE`

Then apply the controller path you want:

```bash
kubectl apply -f k8s/kueue-localqueue.yaml
kubectl apply -f k8s/torchtitan-oke-fss-pvc.yaml
kubectl apply -f k8s/torchtitan-kueue-jobset.yaml
```

```bash
kubectl apply -f k8s/kueue-localqueue.yaml
kubectl apply -f k8s/torchtitan-oke-fss-pvc.yaml
kubectl apply -f k8s/torchtitan-kueue-mpijob.yaml
```

## Operational notes

- Keep `metadata.name`, `spec.network.subdomain`, and the JobSet `JOBSET_NAME` env aligned.
- Keep `replicatedJobs[0].replicas` and `NNODES` aligned in the JobSet path.
- Keep `NPROC_PER_NODE` and `resources.limits["nvidia.com/gpu"]` aligned in the JobSet path.
- Keep `Worker.replicas` and the launcher `WORKER_REPLICAS` env in sync.
- Keep `slotsPerWorker` and `resources.limits["nvidia.com/gpu"]` in sync.
- If your Kueue flavor targets tainted GPU nodes, give the launcher the same GPU toleration as the workers even though it does not request GPUs itself.
- Avoid downloading Hugging Face assets independently in every pod; stage them on the image or the shared FSS volume.
- Do not replace the shared FSS claim with a single `oci-bv` block-volume PVC unless you also redesign the workload so each pod gets its own claim.
- If you need retries, adjust the MPIJob run policy and make sure checkpointing is configured.
- The launcher is intentionally small; only the worker replicas request GPUs, but the container image still needs `mpirun`, `ssh-keygen`, and `sshd` available for the MPI bootstrap path.
