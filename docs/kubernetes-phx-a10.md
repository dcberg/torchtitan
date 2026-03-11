# Running TorchTitan Smoke Tests on the PHX `gpu-a10-2` node pool

This document describes the verified smoke-test path for the current Phoenix OKE A10 pool. It is intentionally not the place for the full `llama3_8b` profile.

This example targets the current Phoenix OKE cluster layout that was validated from the live cluster:

- 3 worker nodes in pool `gpu-a10-2`
- shape `VM.GPU.A10.2`
- `2` GPUs per node
- labels:
  - `oke.oraclecloud.com/pool.name=gpu-a10-2`
  - `node.kubernetes.io/instance-type=VM.GPU.A10.2`

The manifests live in `k8s/phx-a10/`.

## Scope

- `k8s/phx-a10/torchtitan-jobset.yaml` is the PHX smoke-test manifest that was validated successfully on the A10 pool.
- `k8s/phx-a10/torchtitan-jobset-llama3-8b.yaml` is a separate template for a larger GPU pool and is suspended by default.
- Do not point the smoke-test manifest at `llama3_8b`; that profile OOMs on `VM.GPU.A10.2`.

## What the example does

- Creates a `ResourceFlavor` pinned to the PHX GPU node pool.
- Creates a `ClusterQueue` sized to the allocatable resources on the three GPU nodes:
  - `nvidia.com/gpu=6`
  - `cpu=179019m`
  - `memory=1432813104Ki`
- Creates a namespace-scoped `LocalQueue`.
- Creates a `PriorityClass` for batch workloads.
- Launches a `JobSet` with:
  - `replicas=3`
  - `NNODES=3`
  - `NPROC_PER_NODE=2`
  - `CONFIG=llama3_debugmodel` because the full `llama3_8b` profile OOMs on `VM.GPU.A10.2`
  - `--dataloader.dataset c4` so the debug config uses streamed C4 data instead of the repo-local `c4_test` files
  - `HF_ASSETS_PATH=/workspace/tokenizer`, backed by a ConfigMap created from the repo test tokenizer files
  - `nvidia.com/gpu=2` per pod
  - `ephemeral-storage=8Gi` per pod
  - memory-backed `/dev/shm` sized to `8Gi` per pod for NCCL shared-memory transport

With those requests, Kubernetes can place only one TorchTitan pod on each `VM.GPU.A10.2` node.
The example keeps `ephemeral-storage` at `8Gi` per pod because the PHX GPU nodes expose only about `34Gi` of allocatable local ephemeral disk, and the original `30Gi` request caused `DiskPressure` and pod eviction during image pull and container startup.

This smoke-test path was verified end to end in the PHX cluster: all three trainer pods completed and the JobSet reached `Completed`.

## Prerequisites

- Kueue is installed in the cluster.
- JobSet is installed in the cluster.
- The GPU nodes have already run the boot-volume growfs step, so kubelet sees the expanded local filesystem and the higher ephemeral-storage allocatable value.
  - On the current OL8 OKE GPU nodes, that means running the node-level `oci-growfs` helper and then restarting kubelet, or baking the same step into the node pool custom cloud-init so replacement nodes come up with the expanded filesystem automatically.
- The shared PVC from `k8s/phx-a10/torchtitan-oke-fss-pvc.yaml` exists.
- The test-tokenizer ConfigMap exists in `torchtitan-phx`:
  - `kubectl create configmap torchtitan-test-tokenizer -n torchtitan-phx --from-file=tests/assets/tokenizer/tokenizer.json --from-file=tests/assets/tokenizer/tokenizer_config.json`
- The TorchTitan image `phx.ocir.io/odx-mockcustomer/torchtitan:0.0.2` is pushed and reachable from the cluster.
- The GPU node pool is healthy and the NVIDIA operator has finished reconciling.
- The current PHX GPU operator setup exposes `/dev/nvidia*` into GPU pods but does not inject `libcuda.so`, so this example mounts host `/usr/lib64` and wires only the NVIDIA driver libraries into an isolated in-container path before launch.
- The example also mounts an `emptyDir` on `/dev/shm`; the default container shared-memory size is too small for this 6-process NCCL topology and causes `No space left on device` during communicator setup.

Without the growfs step, these nodes can report much lower allocatable ephemeral storage than the boot volume size suggests, which is enough to trigger `DiskPressure` or image-pull/startup failures even for the smoke-test manifest.

If you need to verify or remediate a node manually, the operational pattern is:

```bash
sudo /usr/libexec/oci-growfs -y
sudo systemctl restart kubelet
```

At the time this example was created, the cluster did not yet have the JobSet CRD installed, so you must install JobSet before applying `k8s/phx-a10/torchtitan-jobset.yaml`.

## Apply order

```bash
kubectl apply -f k8s/phx-a10/priorityclass.yaml
kubectl apply -f k8s/phx-a10/resourceflavor.yaml
kubectl apply -f k8s/phx-a10/clusterqueue.yaml
kubectl apply -f k8s/phx-a10/namespace-localqueue.yaml
kubectl apply -f k8s/phx-a10/torchtitan-oke-fss-pvc.yaml
kubectl create configmap torchtitan-test-tokenizer -n torchtitan-phx \
  --from-file=tests/assets/tokenizer/tokenizer.json \
  --from-file=tests/assets/tokenizer/tokenizer_config.json
kubectl apply -f k8s/phx-a10/torchtitan-jobset.yaml
```

## What to customize

- `image` in `k8s/phx-a10/torchtitan-jobset.yaml` if you want to run a different tag or registry
- `MODULE` and `CONFIG`
- `HF_ASSETS_PATH`, `--dump_folder`, and `--checkpoint.folder`
- container CPU, memory, and ephemeral-storage requests if your run needs different sizing
- `ephemeral-storage` should stay comfortably below node allocatable local disk unless you also change the node profile
- PVC storage class and size in `k8s/phx-a10/torchtitan-oke-fss-pvc.yaml`
- the host driver-library workaround if the cluster runtime configuration is fixed later and no longer needs it

If you want to run with the real Llama 3.1 tokenizer instead of the mounted test tokenizer, download the Hugging Face assets, place them somewhere reachable by the pods, and point `HF_ASSETS_PATH` at that directory.

## Full 8B Template

The repository now includes a separate full-profile template:

- `k8s/phx-a10/torchtitan-jobset-llama3-8b.yaml`

That manifest is intentionally different from the smoke-test example:

- It is `suspend: true` by default.
- It expects a larger GPU pool, not `gpu-a10-2`.
- It requests `8` GPUs on one node and assumes materially more memory per GPU than an A10.
- It expects real Llama 3.1 HF assets at `/workspace/assets/hf/Llama-3.1-8B`.
- It keeps the host driver-library workaround and enlarged `/dev/shm`.

Before using the 8B template, update:

- `kueue.x-k8s.io/queue-name`
- `oke.oraclecloud.com/pool.name`
- `node.kubernetes.io/instance-type`
- PVC contents so `/workspace/assets/hf/Llama-3.1-8B` exists
- CPU, memory, and ephemeral-storage requests if your larger shape needs different sizing
- `spec.suspend` from `true` to `false`

## Verify placement

The workload should land only on nodes from the `gpu-a10-2` pool:

```bash
kubectl get pods -n torchtitan-phx -o wide
kubectl get workloads -n torchtitan-phx
kubectl get jobsets -n torchtitan-phx
```

To confirm the GPU operator and Node Feature Discovery have finished on the target nodes:

```bash
kubectl get pods -n gpu-operator -o wide
kubectl get pods -n node-feature-discovery -o wide
kubectl get nodes -l oke.oraclecloud.com/pool.name=gpu-a10-2 --show-labels
```
