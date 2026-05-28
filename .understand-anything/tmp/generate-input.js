const fs = require('fs');
const scanData = JSON.parse(fs.readFileSync('/home/anhuynh/anhc_agent_ws/src/.understand-anything/tmp/ua-scan-files.json', 'utf8'));
const input = {
  projectRoot: '/home/anhuynh/anhc_agent_ws/src',
  files: scanData.files
};
fs.writeFileSync('/home/anhuynh/anhc_agent_ws/src/.understand-anything/tmp/ua-import-map-input.json', JSON.stringify(input));
