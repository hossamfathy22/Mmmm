// Build single-file versions of the engine.
//   node tools/build.mjs [artifactOut]
//   → rasm-standalone.html  (everything inline, works offline: open it on any device)
//   → artifactOut (optional): page content for publishing, opentype.js from jsDelivr
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');
let html = read('index.html');
html = html.replace('<link rel="stylesheet" href="src/app.css">', () => '<style>\n' + read('src/app.css') + '\n</style>');
html = html.replace(/<script src="(src\/[^"]+)"><\/script>/g, (_, p) => '<script>\n' + read(p) + '\n</script>');

const standalone = html.replace(/<!--rasm:lib-->[\s\S]*?<!--\/rasm:lib-->/, () => '<script>\n' + read('vendor/opentype.min.js') + '\n</script>');
fs.writeFileSync(path.join(root, 'rasm-standalone.html'), standalone);
console.log('rasm-standalone.html', (standalone.length / 1024).toFixed(0) + ' KB');

const out = process.argv[2];
if (out) {
  const art = html.replace(/<!--rasm:lib-->[\s\S]*?<!--\/rasm:lib-->/, '<script src="https://cdn.jsdelivr.net/npm/opentype.js@1.3.4/dist/opentype.min.js"></script>');
  const head = art.match(/<head>([\s\S]*?)<\/head>/)[1]
    .replace(/<meta charset[^>]*>\s*/i, '')
    .replace(/<meta name="viewport"[^>]*>\s*/i, '');
  const body = art.match(/<body>([\s\S]*?)<\/body>/)[1];
  fs.writeFileSync(out, head.trim() + '\n' + body.trim() + '\n');
  console.log(out, (fs.statSync(out).size / 1024).toFixed(0) + ' KB');
}
