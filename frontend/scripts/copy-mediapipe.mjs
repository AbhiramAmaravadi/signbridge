// Copies the self-hosted MediaPipe runtime from node_modules into public/mediapipe
// so the browser never has to reach a third-party CDN. Runs before `dev` and `build`.
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const targets = [
  ['@mediapipe/holistic', 'holistic'],
  ['@mediapipe/drawing_utils', 'drawing_utils'],
]

for (const [pkg, name] of targets) {
  const from = join(root, 'node_modules', pkg)
  const to = join(root, 'public', 'mediapipe', name)
  if (!existsSync(from)) {
    console.error(`[copy-mediapipe] ${pkg} is not installed. Run "npm install" first.`)
    process.exit(1)
  }
  if (existsSync(to) && readdirSync(to).length > 0) continue // already copied
  rmSync(to, { recursive: true, force: true })
  mkdirSync(to, { recursive: true })
  cpSync(from, to, {
    recursive: true,
    filter: (src) => !/(package\.json|README\.md|\.d\.ts)$/.test(src),
  })
  console.log(`[copy-mediapipe] ${pkg} -> public/mediapipe/${name}`)
}
