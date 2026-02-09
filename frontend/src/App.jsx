import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Button } from 'antd';
import { HomeOutlined, SettingOutlined, BarChartOutlined, ExperimentOutlined } from '@ant-design/icons';
import Home from './pages/Home';
import ModelConfig from './pages/ModelConfig';
import TrainingDashboard from './pages/TrainingDashboard';
import VQAPlayground from './pages/VQAPlayground';

const { Header, Content, Footer } = Layout;

const Navbar = () => {
    const location = useLocation();

    const items = [
        { key: '/', icon: <HomeOutlined />, label: <Link to="/">Home</Link> },
        { key: '/config', icon: <SettingOutlined />, label: <Link to="/config">Model Config</Link> },
        { key: '/training', icon: <BarChartOutlined />, label: <Link to="/training">Training</Link> },
        { key: '/test', icon: <ExperimentOutlined />, label: <Link to="/test">VQA Test</Link> },
    ];

    return (
        <Header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 50px' }}>
            <div className="logo" style={{ color: '#fff', fontSize: '1.5rem', fontWeight: 'bold', marginRight: '40px' }}>
                <span style={{ color: '#00e5ff' }}>VQA-CL</span> Framework
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
                <Content style={{ padding: '0', display: 'flex', flexDirection: 'column' }}>
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/config" element={<ModelConfig />} />
                        <Route path="/training" element={<TrainingDashboard />} />
                        <Route path="/test" element={<VQAPlayground />} />
                    </Routes>
                </Content>
                <Footer style={{ textAlign: 'center', background: 'transparent', color: '#666' }}>
                    © 2026 VQA-CL Framework. All rights reserved.
                </Footer>
            </Layout>
        </Router>
    );
}

export default App;
