
<!-- /mnt/e/SyncData/paper2_main/zreport/available_dataset.xlsx -->
<!-- ![dataset_list](3.fig/dataset_list.png)
There is a lack of publicly available datasets for indoor fire/smoke detection with static cameras.

Some existing fire/smoke video datasets but using moving cameras like:

- FireNet [@jadon2019firenet]
- Firesense [@Firesens4:online]
- FiSmo [@cazzolato2017fismo]
- FURG [@steffens2015unconstrained]

Static:

- DFire [@dfiredataset]
- VisiFire Blinkent [@VisiFireBilkent:online]
- KMU Fire and Smoke dataset [@KMUFireSmokeDataset]
- Mivia Fire Dataset [@foggia2015real]
- Mivia Smoke Dataset [@foggia2015real]
- USTC Smoke Dataset [@lin2017smoke]

problems:

- mostly outdoor scenes
- small number of videos
- low resolution
- lack of diversity in fire/smoke appearances and environmental conditions

we constructed our own dataset with static cameras in indoor environments to address these limitations.

- combined:
- fire/smoke videos from existing datasets
  - Korea AI Fire Dataset [@AIHub87:online]
  - USTC Smoke Dataset [@lin2017smoke]
  - VSD3K Dataset [@huang2022fire]
  - video collected from the internet (Pexels, pixabay, youtube)
- for non-fire/smoke videos:
  - some self-collected videos from real CCTV cameras in indoor environments (parking areas)
  - Safe&Unsafe behavior in workplaces dataset [@onal2024video]
  - Indoor Action dataset [@deniz2024optimized]
  - MPII Cooking 2 Dataset [@rohrbach2016recognizing]
  - USTC Smoke Dataset [@lin2017smoke]
  - WiseNet dataset [@marroquin2019wisenet]

Statistic of our indoor static video dataset shown in:
`/mnt/e/SyncData/paper2_main/zreport/my_fire_static_indoor_dataset.csv` -->


<!-- !MainPC -->

ha@DESKTOP-JQD9K01
OS: Windows 10 Pro (22H2) x86_64
Kernel: WIN32_NT 10.0.19045.5965
Uptime: 15 days, 13 hours, 1 min
Packages: 17 (scoop), 70 (choco)
CPU: Intel(R) Core(TM) i9-10900K (20) @ 3.70 GHz
GPU: NVIDIA GeForce RTX 3090 (23.76 GiB) [Discrete]
Memory: 36.50 GiB / 63.88 GiB (57%)
Disk (C:\): 396.30 GiB / 476.30 GiB (83%) - NTFS
Disk (D:\): 4.75 TiB / 5.46 TiB (87%) - NTFS
Disk (E:\): 776.22 GiB / 931.50 GiB (83%) - NTFS
Disk (F:\): 66.01 MiB / 10.00 GiB (1%) - NTFS [External]
Disk (G:\): 783.98 GiB / 931.50 GiB (84%) - FAT32
Disk (H:\): 783.98 GiB / 931.50 GiB (84%) - FAT32
Disk (J:\): 729.42 MiB / 3.00 GiB (24%) - NTFS [External]
Local IP (vEthernet (Internet Switch)): 115.145.67.115/24

<!-- !1GPU server

<!-- comeduTa1@DESKTOP-QNS3DNF
OS: Windows 10 Pro (21H2) x86_64
Kernel: WIN32_NT 10.0.19044.3086
Uptime: 23 days, 2 hours, 16 mins
Packages: 45 (choco)
CPU: 12th Gen Intel(R) Core(TM) i9-12900K (24) @ 3.19 GHz
GPU 1: Microsoft Remote Display Adapter
GPU 2: NVIDIA GeForce RTX 3090 (23.76 GiB) [Discrete]
Memory: 17.99 GiB / 63.75 GiB (28%)
Disk (C:\): 1.10 TiB / 1.82 TiB (61%) - NTFS
Disk (D:\): 1.43 TiB / 1.82 TiB (79%) - NTFS
Disk (E:\): 1.40 TiB / 7.28 TiB (19%) - NTFS
Local IP (115.145.36.213/24)
NVIDIA-SMI 575.51.02              Driver Version: 576.02         CUDA Version: 12.9
torch 2.7.1+ cu118 -->


The experiments were conducted using a Windows 10 Pro 22H2 operating system with 64 GB of
RAM and two NVIDIA GeForce RTX 3090 GPUs, each with 24 GB of VRAM. The NVIDIA CUDA
Toolkit version 11.8 was used in conjunction with PyTorch 2.1.0 for training YOLO models. -->

<!-- !4GPU server -->

comeduta5@DESKTOP-Q2IKLC0
OS: Windows 10 Pro (22H2) x86_64
Kernel: WIN32_NT 10.0.19045.6456
Uptime: 92 days, 5 hours, 58 mins
Packages: 35 (choco)
CPU: 2 x Intel(R) Xeon(R) Silver 4210R (40) @ 4.00 GHz
GPU 1: NVIDIA GeForce RTX 3090 (23.76 GiB) [Discrete]
GPU 2: Microsoft Remote Display Adapter
GPU 3: Microsoft Basic Display Adapter [Integrated]
GPU 4: NVIDIA GeForce RTX 3090 (23.76 GiB) [Discrete]
GPU 5: NVIDIA GeForce RTX 3090 (23.76 GiB) [Discrete]
GPU 6: NVIDIA GeForce RTX 3090 (23.76 GiB) [Discrete]
Memory: 24.69 GiB / 127.63 GiB (19%)
Disk (C:\): 574.13 GiB / 975.92 GiB (59%) - NTFS
Disk (D:\): 260.78 GiB / 446.62 GiB (58%) - NTFS
Disk (E:\): 898.97 GiB / 1.23 TiB (71%) - NTFS
Local IP (NIC1): 115.145.36.212/24

All inference latencies were measured on an NVIDIA Jetson Nano / RTX 3060 to simulate edge deployment
