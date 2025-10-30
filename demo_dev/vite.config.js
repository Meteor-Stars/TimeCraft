// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'
// import { ReactInjectorVitePlugin } from 'yunji-tagger'
//
// export default defineConfig({
//   plugins: [react(), ReactInjectorVitePlugin()],
// })

// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'
// import { ReactInjectorVitePlugin } from 'yunji-tagger'
//
// export default defineConfig({
//   plugins: [react(), ReactInjectorVitePlugin()],
//   server: {
//     host: '0.0.0.0', // ✅ 监听所有网络接口
//     port: 5173,      // ✅ 明确指定端口
//     strictPort: true,// 如果端口被占用直接退出
//     // 可选：如果你有反向代理或特殊需求
//     // hmr: {
//     //   clientPort: 5173,
//     //   host: 'localhost'
//     // }
//   }
// })


// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'
// import { ReactInjectorVitePlugin } from 'yunji-tagger'
//
// export default defineConfig({
//   plugins: [react(), ReactInjectorVitePlugin()],
//   server: {
//     host: '0.0.0.0',        // 监听所有接口
//     port: 5173,             // 指定端口
//     strictPort: true,       // 端口占用时直接退出
//     // ✅ 关键：设置为 '0.0.0.0' 后，Vite 会自动信任所有 host
//   }
// })


import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { ReactInjectorVitePlugin } from 'yunji-tagger'

export default defineConfig({
  plugins: [react(), ReactInjectorVitePlugin()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // ✅ 显式允许 ngrok 域名
    allowedHosts: [
      'ununified-outrightly-nickole.ngrok-free.dev',
      // 如果你以后用其他 ngrok 链接，也要加进去
    ]
  }
})