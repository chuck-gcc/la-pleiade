⚙️ Fiche de configuration VS Code – Projet Pleiades. 
🧩 Objectif. 
  
Configurer Visual Studio Code pour :  
  
travailler dans un environnement Python virtuel (merope),  

accéder aux bibliothèques compilées localement dans Atlas (ROCm SDK),

faire reconnaître ces chemins par Pylance (analyse statique),

exécuter et déboguer des scripts directement sur le serveur GPU.

📦 Pré-requis
Élément	Description	Vérification
VS Code	Éditeur principal	code --version
Extension Python	Analyse & exécution	installée depuis le marketplace
(Optionnel) Remote – SSH	Connexion à ton serveur GPU	Ctrl+Shift+P → Remote-SSH: Connect to Host
Python virtuel merope	Environnement du projet	source merope/bin/activate
Build ROCm atlas	SDK compilé localement	../atlas/build/dist/rocm/ existe
🧠 Structure de ton projet
pleiades/
├── atlas/        # ROCm SDK compilé avec TheRock
│   └── build/dist/rocm/
│       ├── bin/                 → binaires ROCm (hipcc, amd-smi…)
│       ├── lib/                 → bibliothèques (.so)
│       └── share/amd_smi/       → module Python amdsmi
├── merope/       # environnement Python
│   ├── bin/activate
│   ├── bridge_rocm.sh
│   └── notebooks/
└── .vscode/
    ├── settings.json
    ├── launch.json
    └── tasks.json

🔧 1️⃣ Configuration principale (.vscode/settings.json)

Ce fichier indique à VS Code quel Python utiliser et où chercher les modules :

{
  // Sélection du venv Python de Merope
  "python.defaultInterpreterPath": "/home/cc/gpu_lab/pleiades/merope/bin/python",

  // Ajout des bibliothèques ROCm compilées localement
  "python.analysis.extraPaths": [
    "/home/cc/gpu_lab/pleiades/atlas/build/dist/rocm/share/amd_smi"
  ],

  // Optionnel : activation de l’auto-détection
  "python.analysis.autoSearchPaths": true,
  "python.analysis.useLibraryCodeForTypes": true,

  // Formatage et style
  "editor.formatOnSave": true,
  "python.formatting.provider": "black",

  // Terminal intégré
  "terminal.integrated.env.linux": {
    "LD_LIBRARY_PATH": "/home/cc/gpu_lab/pleiades/atlas/build/dist/rocm/lib:${env:LD_LIBRARY_PATH}",
    "PYTHONPATH": "/home/cc/gpu_lab/pleiades/atlas/build/dist/rocm/share/amd_smi:${env:PYTHONPATH}",
    "PATH": "/home/cc/gpu_lab/pleiades/atlas/build/dist/rocm/bin:${env:PATH}"
  }
}


💡 Astuce : ces variables permettent au terminal intégré et à Pylance d’avoir la même visibilité que ton script bridge_rocm.sh.

🧩 2️⃣ Débogage (.vscode/launch.json)

Permet d’exécuter ou de déboguer ton code directement depuis VS Code :

{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Merope script",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "LD_LIBRARY_PATH": "/home/cc/gpu_lab/pleiades/atlas/build/dist/rocm/lib",
        "PYTHONPATH": "/home/cc/gpu_lab/pleiades/atlas/build/dist/rocm/share/amd_smi"
      }
    }
  ]
}


➡️ Clique simplement sur ▶️ dans VS Code pour lancer ton script GPU.

🧱 3️⃣ Tâches de build (.vscode/tasks.json)

Utile si tu veux compiler du code HIP/C++ directement depuis VS Code :

{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build HIP",
      "type": "shell",
      "command": "hipcc ${file} -o ${fileDirname}/${fileBasenameNoExtension}",
      "group": "build",
      "problemMatcher": []
    }
  ]
}

💻 4️⃣ Utilisation avec Remote – SSH

Quand tu travailles sur le serveur GPU :

Installe l’extension Remote – SSH dans VS Code.

Connecte-toi :
Ctrl + Shift + P → Remote-SSH: Connect to Host

Ouvre ton dossier distant /home/cc/gpu_lab/pleiades/

Re-sélectionne ton interpréteur :
Ctrl + Shift + P → Python: Select Interpreter → merope/bin/python

Recharge la fenêtre (Developer: Reload Window)

🧠 5️⃣ Bonnes pratiques
Objectif	Bonne pratique
Cohérence entre terminal et VS Code	toujours exécuter source bridge_rocm.sh avant d’ouvrir VS Code
Analyser sans erreur les modules custom (amdsmi, hip)	utiliser python.analysis.extraPaths
Utiliser PyTorch ROCm	vérifier que torch.version.hip retourne une version valide
Sauvegarde automatique des notebooks	activer jupyter.autosave.enabled: true
Gestion de projet propre	ajouter .vscode/ à ton .gitignore si tu veux garder tes réglages personnels
🔍 6️⃣ Vérification rapide

Tu peux créer un fichier check_env.py :

import torch, amdsmi, os

print("PyTorch:", torch.__version__)
print("ROCm:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("amdsmi:", amdsmi.amdsmi_get_lib_version())
print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"))


Exécute-le via VS Code → ▶️
Si tout s’affiche correctement, ta config est 100 % opérationnelle ✅

📘 7️⃣ Résumé express
Élément	Fichier	Contenu clé
Interpréteur Python	.vscode/settings.json	"python.defaultInterpreterPath"
ROCm extraPaths	.vscode/settings.json	"python.analysis.extraPaths"
Débogage GPU	.vscode/launch.json	"env": {"LD_LIBRARY_PATH": …}
Compilation HIP	.vscode/tasks.json	hipcc ${file}
Terminal cohérent	bridge_rocm.sh	export des chemins ROCm
Remote SSH	Config VS Code	interpréteur distant sélectionné