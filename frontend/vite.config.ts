import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rolldownOptions: {
      output: {
        // Third-party code changes rarely: keep it in its own long-cached chunks.
        codeSplitting: {
          groups: [
            { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
            { name: 'motion', test: /node_modules[\\/](framer-motion|motion-dom|motion-utils)[\\/]/ },
            // GSAP is left out so it stays with the lazy Features page.
            { name: 'vendor', test: /node_modules[\\/](?!gsap|@gsap)/ },
          ],
        },
      },
    },
  },
})
