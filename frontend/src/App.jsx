import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import { HomeOutlined, AppstoreOutlined, FileTextOutlined, ExperimentOutlined } from '@ant-design/icons';
import Home from './components/Home';
import Models from './components/Models';
import ModelDocs from './components/ModelDocs';
import VQAPlayground from './components/VQAPlayground';

const { Header, Content, Footer } = Layout;

const Navbar = () => {
    const location = useLocation();

    const items = [
        { key: '/', icon: <HomeOutlined />, label: <Link to="/">Home</Link> },
        { key: '/models', icon: <AppstoreOutlined />, label: <Link to="/models">Models</Link> },
        { key: '/docs', icon: <FileTextOutlined />, label: <Link to="/docs">Documentation</Link> },
        { key: '/test', icon: <ExperimentOutlined />, label: <Link to="/test">VQA Playground</Link> },
    ];

    return (
        <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 50px' }}>
            <div className="logo" style={{ color: '#fff', fontSize: '1.5rem', fontWeight: 'bold', marginRight: '40px' }}>
                <span style={{ color: '#00e5ff' }}>DyCDC</span> Training Strategy
            </div>
            <Menu
                theme="dark"
                mode="horizontal"
                selectedKeys={[location.pathname]}
                items={items}
                style={{ flex: 1, minWidth: 0, justifyContent: 'flex-end' }}
            />
        </Header>
    );
};

function App() {
    return (
        <Router>
            <Layout style={{ minHeight: '100vh', background: 'transparent' }}>
                <Navbar />
                <Content style={{ padding: '0' }}>
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/models" element={<Models />} />
                        <Route path="/docs" element={<ModelDocs />} />
                        <Route path="/test" element={<VQAPlayground />} />
                    </Routes>
                </Content>
                <Footer style={{ textAlign: 'center', background: 'transparent', color: '#666' }}>
                    2026 DyCDC Training Strategy. All rights reserved.
                </Footer>
            </Layout>
        </Router>
    );
}

export default App;
