🚀 Fiche complète — Installation et tests de TheRock + ROCm
🧭 Contexte

Tu travailles dans une arborescence structurée :

pleiades/
├── atlas/      ← build et SDK ROCm via TheRock
├── merope/     ← environnement Python / PyTorch ROCm
├── electra/    ← environnements CUDA/NVIDIA
├── alcyone/    ← outils CPU / debug
└── docker/     ← images Docker CPU/ROCm/CUDA


Ton GPU principal :
AMD Radeon PRO W7900 Dual Slot (gfx1100)
Architecture RDNA3 compatible ROCm 7.x.

🧱 1️⃣ Installation de base : ROCm Core SDK avec TheRock
📦 Prérequis système

Dans ton container ou ta machine :

apt update
apt install -y automake cmake g++ gfortran git git-lfs \
  libegl1-mesa-dev libtool ninja-build patchelf pip \
  pkg-config python3-dev python3-venv xxd

🧰 Préparer un environnement Python isolé
cd pleiades/atlas
python3 -m venv .rockenv
source .rockenv/bin/activate

🪨 Cloner et configurer TheRock
git clone https://github.com/ROCm/TheRock.git
cd TheRock
pip install -r requirements.txt

📥 Télécharger les sources ROCm (librairies et outils)
python build_tools/fetch_sources.py


⏳ Télécharge ~16–17 Go (rocBLAS, MIOpen, HIP, rocRAND, etc.)

⚙️ Compiler avec CMake + Ninja

Définis ta cible GPU (ex. gfx1100 pour W7900) :

cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_TARGETS=gfx1100 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_BLAS=ON \
  -DTHEROCK_ENABLE_ML_LIBS=ON


Lance la compilation :

cmake --build build

🧪 Vérification de la build

Test basique :

ctest --test-dir build


Attendu :

100% tests passed, 0 tests failed out of 24

📂 Structure de sortie

Une fois le build terminé, les binaires sont ici :

pleiades/atlas/build/dist/rocm/
├── bin/          → rocm-smi, rocminfo, hipcc, etc.
├── lib/          → libamdhip64.so, librocblas.so, libMIOpen.so...
├── include/      → headers HIP et ROCm
└── share/        → scripts et doc

🧩 2️⃣ Test du runtime ROCm (niveau système)

Vérifie la détection du GPU :

build/dist/rocm/bin/rocminfo | grep Name


Attendu :

Name: gfx1100


Vérifie la gestion de la carte :

build/dist/rocm/bin/rocm-smi


Affiche température, charge, mémoire, puissance, etc.

🧠 3️⃣ Intégration à ton environnement Python (Merope)
🧰 Créer ton venv Python
cd pleiades/merope
python3 -m venv venv
source venv/bin/activate

🔗 Lier Merope ↔ Atlas (accès au ROCm build local)

Crée scripts/activate_rocm.sh :

#!/bin/bash
ATLAS_ROOT="$(realpath ../atlas)"
ROCM_BIN="$ATLAS_ROOT/build/dist/rocm/bin"
ROCM_LIB="$ATLAS_ROOT/build/dist/rocm/lib"

export PATH="$ROCM_BIN:$PATH"
export LD_LIBRARY_PATH="$ROCM_LIB:$LD_LIBRARY_PATH"

echo "[Merope] Connected to ROCm from Atlas."


Et rends-le exécutable :

chmod +x scripts/activate_rocm.sh


À chaque session :

source scripts/activate_rocm.sh

🔥 4️⃣ Installation de PyTorch ROCm

Désinstalle les versions CUDA :

pip uninstall torch torchvision torchaudio -y


Puis installe la version ROCm officielle :

pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0


Vérifie :

python -c "import torch; print(torch.__version__, torch.version.hip)"


Résultat attendu :

2.9.0+rocm7.0 7.0.2

🧮 5️⃣ Test de calcul GPU : test_gpu.py
import torch
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
a = torch.randn((4096, 4096), device="cuda")
b = torch.randn((4096, 4096), device="cuda")
c = torch.matmul(a, b)
print("Done:", c.sum().item())


✅ Résultat attendu :

GPU: AMD Radeon PRO W7900 Dual Slot
Done: <valeur>

📊 6️⃣ Test de performance et logs : test_gpu_verbose.py

Script de stress avec logs GPU (rocm-smi) :

Alloue des matrices (16k × 16k ou 32k × 32k),

Effectue un matmul sur GPU,

Affiche température, puissance, mémoire,

Mesure le temps exact du kernel.

python test_gpu_verbose.py


Sortie attendue :

GPU use (%) : 99
Power (W): 500+
Matrix multiplication done on GPU!

🧾 7️⃣ Vérifications supplémentaires
Test	Commande	Résultat attendu
Détection GPU	`rocminfo	grep Name`
Charge GPU	rocm-smi --showuse	GPU use (%) > 90
Libs ROCm liées à PyTorch	`ldd $(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.file), 'lib/libtorch_hip.so'))")	grep rocm`
Variable HIP	echo $HIP_VISIBLE_DEVICES	Doit être 0 ou vide
Backend PyTorch	torch.version.hip	Non None
⚡ 8️⃣ Optimisation (pour saturer le GPU)

Pour forcer une charge à 99 % :

N = 32768
for i in range(5):
    a = torch.randn((N, N), dtype=torch.float16, device="cuda")
    b = torch.randn((N, N), dtype=torch.float16, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()


Surveille en parallèle :

watch -n 0.5 rocm-smi --showuse --showpower --showtemp

✅ 9️⃣ Résumé des points de validation
Étape	Objectif	Outil / commande	Résultat attendu
Build TheRock	Compiler ROCm SDK local	cmake --build build	dist/rocm créé
ROCm visible	GPU détecté	rocminfo	gfx1100
Environnement Merope	Activation du PATH ROCm	source scripts/activate_rocm.sh	Chemins exportés
PyTorch ROCm	Framework prêt	torch.version.hip	7.x
Matmul test	Calcul GPU	python test_gpu.py	Résultat numérique
GPU load	Vérifier activité	rocm-smi	use (%) ~99
📦 10️⃣ Nettoyage / Maintenance

Pour reconstruire ROCm :

cd pleiades/atlas/TheRock
rm -rf build
cmake -B build -GNinja .
cmake --build build


Pour mettre à jour les sources ROCm :

python build_tools/fetch_sources.py --update

🧠 En résumé
Composant	Rôle
TheRock	Système de build CMake/Ninja unifié pour ROCm
ROCm SDK (Atlas)	Librairies HIP, rocBLAS, MIOpen, etc.
Merope	Environnement Python/PyTorch ROCm
activate_rocm.sh	Pont entre Merope et Atlas
PyTorch ROCm wheel	Backend HIP pour Python
rocm-smi / rocminfo	Outils de diagnostic GPU
test_gpu_verbose.py	Benchmark et monitoring GPU
