import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function replaceColors(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const fullPath = path.join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
            replaceColors(fullPath);
        } else if (fullPath.endsWith('.jsx')) {
            let content = fs.readFileSync(fullPath, 'utf8');
            let modified = false;

            const colorMap = {
                '#f8fafc': '#ffffff',
                '#f1f5f9': '#f1f4f9',
                '#e2e8f0': '#d1d9e6',
                '#cbd5e1': '#b8c5d9',
                '#64748b': '#475569'
            };

            for (const [before, after] of Object.entries(colorMap)) {
                const regex = new RegExp(before, 'gi');
                if (regex.test(content)) {
                    content = content.replace(regex, after);
                    modified = true;
                }
            }

            if (modified) {
                fs.writeFileSync(fullPath, content, 'utf8');
                console.log(`Updated colors in ${fullPath}`);
            }
        }
    }
}

replaceColors(path.join(__dirname, 'src'));
console.log('Color substitution complete');
