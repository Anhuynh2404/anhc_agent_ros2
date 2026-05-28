const fs = require('fs');
const scanData = JSON.parse(fs.readFileSync('/home/anhuynh/anhc_agent_ws/src/.understand-anything/tmp/ua-scan-files.json', 'utf8'));
const importMapData = JSON.parse(fs.readFileSync('/home/anhuynh/anhc_agent_ws/src/.understand-anything/tmp/ua-import-map-output.json', 'utf8'));

const languages = Array.from(new Set(Object.keys(scanData.stats.byLanguage))).sort();

const description = "ANHCAPE is a customized Deep Reinforcement Learning architecture for mobile robot navigation using ROS 2 and Gazebo. Built on TD3, it features adaptive exploration, Actor-Critic policy optimization, and a Dual Encoder embedding system for state representation.";

const finalResult = {
  name: "ANHCAPE",
  description: scanData.totalFiles > 100 ? description + " Note: this project has over 100 source files; consider scoping analysis to a subdirectory for faster results." : description,
  languages: languages,
  frameworks: ["Docker", "Gazebo", "PyTorch", "ROS 2"],
  files: scanData.files,
  totalFiles: scanData.totalFiles,
  filteredByIgnore: scanData.filteredByIgnore,
  estimatedComplexity: scanData.estimatedComplexity,
  importMap: importMapData.importMap
};

fs.mkdirSync('/home/anhuynh/anhc_agent_ws/src/.understand-anything/intermediate', { recursive: true });
fs.writeFileSync('/home/anhuynh/anhc_agent_ws/src/.understand-anything/intermediate/scan-result.json', JSON.stringify(finalResult, null, 2));
