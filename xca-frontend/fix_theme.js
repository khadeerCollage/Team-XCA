import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function replaceFonts(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const fullPath = path.join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
            replaceFonts(fullPath);
        } else if (fullPath.endsWith('.jsx') || fullPath.endsWith('.css') || fullPath.endsWith('.js')) {
            let content = fs.readFileSync(fullPath, 'utf8');
            let modified = false;

            // JS / JSX string regex replacements
            const regexInter1 = /'Inter',\s*-apple-system,\s*BlinkMacSystemFont,\s*'Segoe UI',\s*sans-serif/g;
            if (regexInter1.test(content)) { content = content.replace(regexInter1, "'Source Sans 3', sans-serif"); modified = true; }

            const regexInter2 = /'Inter',\s*sans-serif/g;
            if (regexInter2.test(content)) { content = content.replace(regexInter2, "'Source Sans 3', sans-serif"); modified = true; }

            const regexJet1 = /'JetBrains Mono',\s*monospace/g;
            if (regexJet1.test(content)) { content = content.replace(regexJet1, "'IBM Plex Mono', monospace"); modified = true; }

            const regexJet2 = /'JetBrains Mono',\s*'Fira Code',\s*monospace/g;
            if (regexJet2.test(content)) { content = content.replace(regexJet2, "'IBM Plex Mono', monospace"); modified = true; }

            // CSS syntax string replacements
            const regexInterCSS = /'Inter',\s*Arial,\s*sans-serif/g;
            if (regexInterCSS.test(content)) { content = content.replace(regexInterCSS, "'Source Sans 3', Arial, sans-serif"); modified = true; }

            if (modified) {
                fs.writeFileSync(fullPath, content, 'utf8');
                console.log(`Updated fonts in ${fullPath}`);
            }
        }
    }
}

replaceFonts(path.join(__dirname, 'src'));
console.log('Font substitution complete');
