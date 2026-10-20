import RagPanel from './RagPanel'
import MetricsBar from './MetricsBar'
import './App.css'

function App() {
  return (
    <div className="app-shell">
      <main className="app-main">
        {/* 顶部横幅 */}
        <header className="hero rise">
          <div className="hero-content">
            <span className="hero-badge">
              🔍 RAG 知识库
            </span>
            <h1>RAG 知识库问答助手</h1>
            <p>上传 PDF、Word、TXT 等文档，用自然语言提问，由大模型智能检索知识库内容并给出准确回答。支持多轮对话和文献溯源。</p>
          </div>
        </header>

        <RagPanel />
        <MetricsBar />
      </main>
    </div>
  )
}

export default App
