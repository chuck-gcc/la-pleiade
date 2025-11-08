#PREPARATION

cmake_minimum_required()	Version minimale de CMake
project()	Déclare le projet
add_executable()	Crée un binaire
add_library()	Crée une lib statique/dynamique
target_link_libraries()	Lie les dépendances
target_include_directories()	Définit les chemins d’includes
target_compile_options()	Définit les flags du compilateur
set()	Variables ou options de build
file(GLOB ...)	Collecte de fichiers source
add_subdirectory()	Inclusion de sous-projets
install()	Installation système

#BUILD

cmake -S . -B build	Configure le projet (génère les fichiers Make)
cmake --build build	Compile le projet
cmake --install build	Installe les binaires dans /usr/local
cmake --build build --target clean	Nettoie
cmake -DCMAKE_BUILD_TYPE=Release ..	Change le mode de compilation
ccmake . ou cmake-gui	Interface graphique de configuration