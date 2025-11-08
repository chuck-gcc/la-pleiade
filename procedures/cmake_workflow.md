# 🧱 CMake — Fiche complète & Workflow dans un projet comme TheRock

## 🧭 Introduction

**CMake** (Cross-Platform Make) est un outil de génération de projets et de build multiplateforme.  
Il ne compile pas directement le code, mais génère des fichiers pour un **système de build** (comme `ninja`, `make`, `msbuild`, etc.) adaptés à ton environnement.

CMake est aujourd’hui la base de la plupart des projets C++ modernes — y compris **ROCm**, **TheRock**, **LLVM**, **PyTorch**, etc.

---

## ⚙️ 1️⃣ Concept de base

CMake repose sur trois couches :

Source code ─┐
CMakeLists.txt ──► [Configuration] ─► Build system (Make/Ninja)
│
▼
[Build] ─► Compilation / linking

yaml
Copier le code

### Les fichiers clés :
- `CMakeLists.txt` → fichier principal de configuration du projet.
- `CMakeCache.txt` → options sauvegardées (cibles GPU, flags...).
- `build/` → dossier généré contenant les fichiers du build (Makefile, .ninja, etc.).
- `CMakePresets.json` (optionnel) → profils de build (Debug, Release, etc.).

---

## 🧩 2️⃣ Commandes fondamentales

| Commande | Rôle |
|:--|:--|
| `cmake -B build` | Configure le projet (analyse les CMakeLists.txt) |
| `cmake --build build` | Compile le projet avec le backend choisi |
| `ctest --test-dir build` | Lance les tests définis dans CMake |
| `cmake --install build` | Installe les binaires / libs dans le répertoire cible |
| `cmake -LH` | Affiche les options et variables disponibles |
| `ccmake .` ou `cmake-gui .` | Interface interactive pour configurer le build |

---

## 🧠 3️⃣ Structure typique d’un projet CMake (exemple : TheRock)

TheRock/
├── CMakeLists.txt ← projet principal
├── cmake/ ← modules et macros CMake
├── build/ ← répertoire de génération
├── src/ ← code source (C/C++/HIP)
├── include/ ← headers
├── tests/ ← tests unitaires
└── build_tools/ ← scripts de build (fetch_sources.py, setup_ccache.py)

scss
Copier le code

### Exemple de haut niveau (CMakeLists.txt)
```cmake
cmake_minimum_required(VERSION 3.21)
project(TheRock LANGUAGES C CXX)

# Définit le standard
set(CMAKE_CXX_STANDARD 17)

# Active les sous-modules
add_subdirectory(src/hip)
add_subdirectory(src/rocblas)

# Active les tests
enable_testing()
add_subdirectory(tests)
🔩 4️⃣ Cycle de vie d’un build CMake
Étape 1 : Configuration
CMake analyse tous les CMakeLists.txt,
résout les dépendances, génère les fichiers de build.

Exemple :

bash
Copier le code
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_TARGETS=gfx1100 \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_BLAS=ON
🧠 Ici :

-B build → répertoire de sortie.

-GNinja → utilise Ninja au lieu de Make.

Les -D... sont des variables définies par le projet.

🔹 Cette étape ne compile rien, elle configure seulement le build.

Étape 2 : Compilation
CMake appelle le backend (ninja, make, etc.) :

bash
Copier le code
cmake --build build -j 16
Cela compile tous les fichiers .cpp → .o, puis lie les binaires et les libs :

Copier le code
libamdhip64.so
librocblas.so
hipcc
rocm-smi
Étape 3 : Tests
Les tests sont déclarés dans les CMakeLists :

c
Copier le code
add_test(NAME rocblas_test COMMAND rocblas_test_exe)
Puis exécutés :

bash
Copier le code
ctest --test-dir build
Sortie attendue :

matlab
Copier le code
100% tests passed, 0 tests failed out of 24
Étape 4 : Installation
Installe le SDK ROCm compilé dans dist/ :

bash
Copier le code
cmake --install build --prefix dist/rocm
Tu obtiens :

pgsql
Copier le code
dist/rocm/
├── bin/
├── lib/
├── include/
└── share/
🧱 5️⃣ CMake et TheRock
TheRock utilise CMake comme “super chef d’orchestre” pour :

rassembler les sous-projets ROCm (HIP, rocBLAS, MIOpen…),

propager les options GPU (gfx1100, gfx942, etc.),

construire tout le SDK en un seul build cohérent.

Structure interne simplifiée :
csharp
Copier le code
TheRock/
├── base/        ← runtime et outils (HIP)
├── math-libs/   ← BLAS, RAND, SOLVER, etc.
├── ml-libs/     ← MIOpen, hipDNN
└── profiler/    ← rocprofiler, rocminfo
Chaque sous-répertoire contient son propre CMakeLists.txt,
mais la configuration globale (les flags THEROCK_ENABLE_*)
est gérée au niveau du CMakeLists.txt racine.

⚙️ 6️⃣ Variables et options utiles dans TheRock
Variable	Description
THEROCK_AMDGPU_TARGETS	Définit la génération GPU (ex: gfx1100, gfx942)
THEROCK_ENABLE_ALL	Active ou désactive tous les modules
THEROCK_ENABLE_HIP_RUNTIME	Compile le runtime HIP
THEROCK_ENABLE_BLAS	Compile les libs mathématiques (rocBLAS, rocSOLVER…)
THEROCK_ENABLE_ML_LIBS	Compile les libs AI (MIOpen, hipDNN)
THEROCK_ENABLE_MPI	Active le support MPI
CMAKE_BUILD_TYPE	Release, Debug, ou RelWithDebInfo
CMAKE_INSTALL_PREFIX	Dossier d’installation final (dist/rocm/)

🧩 7️⃣ Gestion des dépendances
TheRock utilise :

CMake FetchContent pour cloner les sous-modules.

Python + git (fetch_sources.py) pour récupérer les projets ROCm depuis GitHub.

CMake gère ensuite les dépendances entre libs :

nginx
Copier le code
rocBLAS → rocSOLVER → hipBLAS → HIP runtime → ROCr
Chaque sous-module déclare ses dépendances via target_link_libraries().

🔁 8️⃣ Workflow de travail typique
🔹 Premier build complet
bash
Copier le code
git clone https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_tools/fetch_sources.py

cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_TARGETS=gfx1100 \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_BLAS=ON

cmake --build build -j 16
cmake --install build --prefix dist/rocm
🔹 Rebuild après modification
bash
Copier le code
cmake --build build -j 16
ctest --test-dir build
CMake reconstruit seulement ce qui a changé.

🔹 Clean / rebuild complet
bash
Copier le code
rm -rf build
cmake -B build -GNinja .
cmake --build build
🔹 Utilisation de ccache pour accélérer les rebuilds
bash
Copier le code
sudo apt install ccache
python build_tools/setup_ccache.py
Ensuite :

bash
Copier le code
export CCACHE_DIR=.ccache
export CC="ccache gcc"
export CXX="ccache g++"
🔹 Mode debug
bash
Copier le code
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug
🧠 9️⃣ CMake et CI/CD (Continuous Integration)
Dans un workflow GitHub Actions, un job typique TheRock :

yaml
Copier le code
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: sudo apt install ninja-build cmake g++
      - name: Configure
        run: cmake -B build -GNinja . -DTHEROCK_ENABLE_ALL=ON
      - name: Build
        run: cmake --build build -j 16
      - name: Test
        run: ctest --test-dir build --output-on-failure
CMake fournit une base commune qui s’exécute aussi bien sur :

Linux x86_64

Windows (MSBuild)

MacOS (clang + make)

🧾 10️⃣ Bonnes pratiques
Bonne pratique	Description
Séparer code / build (build/ folder)	Évite de polluer la source
Utiliser -B et -S	-B = build dir, -S = source dir
Toujours configurer avant de build	cmake -B build avant --build
Gérer les options via -D	Évite de modifier les CMakeLists.txt
Nettoyer les caches en cas de bug	rm -rf build
Utiliser Ninja	Plus rapide et clair que Make
Lire CMakeCache.txt	Contient les paramètres effectifs du build

📚 11️⃣ Ressources utiles
📘 Documentation CMake officielle

🪨 ROCm TheRock GitHub

⚙️ ROCm Build Overview

🐍 FetchContent & ExternalProject

💡 Modern CMake Examples

✅ 12️⃣ Résumé
Niveau	Rôle	Exemple
CMake configure	Analyse le projet, prépare les builds	cmake -B build -GNinja .
CMake build	Compile et link	cmake --build build
CTest	Lance les tests	ctest --test-dir build
Install	Crée le SDK final	cmake --install build --prefix dist/rocm
TheRock	Super-projet ROCm piloté par CMake	orchestre hipBLAS, MIOpen, ROCr…

🔹 En résumé :
CMake = cerveau du build
Ninja = bras exécutant
TheRock = squelette du ROCm SDK
Atlas = ton build local ROCm
Merope = ton espace Python/PyTorch qui en tire parti
