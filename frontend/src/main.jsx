import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import './index.css'
import App from './App.jsx'

// 统一视觉风格：极简科技风（冷色灰蓝 + 电光蓝高亮；仅主题层，不影响业务逻辑）
const theme = {
  token: {
    colorPrimary: '#0ea5e9',
    colorInfo: '#0ea5e9',
    colorLink: '#0284c7',
    colorLinkHover: '#0ea5e9',
    colorTextHeading: '#0f172e',
    colorText: '#364156',
    colorTextSecondary: '#687488',
    colorTextTertiary: '#94a3b8',
    colorBorder: '#cbd5e1',
    colorBorderSecondary: '#e8eef5',
    borderRadius: 6,
    fontSize: 14,
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif",
    boxShadowTertiary: '0 1px 3px rgba(15, 23, 46, 0.06)',
  },
  components: {
    Card: {
      borderRadiusLG: 8,
      paddingLG: 20,
      headerFontSize: 15,
      headerBg: 'transparent',
    },
    Table: {
      headerBg: '#eef2f7',
      headerColor: '#364156',
      headerSplitColor: 'transparent',
      borderColor: '#e8eef5',
      rowHoverBg: '#f1f5f9',
    },
    Button: {
      fontWeight: 500,
      primaryShadow: 'none',
    },
    Segmented: {
      trackBg: '#e8eef5',
      itemSelectedBg: '#0f172e',
      itemSelectedColor: '#ffffff',
      itemHoverBg: '#cbd5e1',
    },
    Tabs: {
      itemSelectedColor: '#0284c7',
      itemHoverColor: '#0ea5e9',
      inkBarColor: '#0ea5e9',
      horizontalItemGutter: 24,
    },
    Progress: {
      defaultColor: '#0ea5e9',
      remainingColor: '#e8eef5',
    },
    Empty: {
      colorTextDescription: '#94a3b8',
    },
    Tag: {
      defaultBg: '#e8eef5',
      defaultColor: '#364156',
    },
  },
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ConfigProvider locale={zhCN} theme={theme}>
      <App />
    </ConfigProvider>
  </StrictMode>,
)
