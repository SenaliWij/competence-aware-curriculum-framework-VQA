import React, { useState } from 'react';
import { Row, Col, Card, Typography, Button, Tag, Drawer, Alert, Space, Progress } from 'antd';
import { LockOutlined, InfoCircleOutlined, DownloadOutlined, ExperimentOutlined, FileTextOutlined, ThunderboltOutlined, HddOutlined, CalendarOutlined, BranchesOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { getModelDownloadUrl } from '../services/api';

import viltImg from '../assets/vilt.png';
import lxmertImg from '../assets/lxmert.png';
import oscarImg from '../assets/oscar.png';

const { Title, Paragraph, Text } = Typography;

const mockedModels = [
    {
        id: 'vilt',
        name: 'ViLT-Enhanced',
        type: 'Transformer',
        version: 'v1.0',
        description: 'Vision-and-Language Transformer optimized with curriculum learning.',
        locked: false,
        size: '1.3GB',
        date: '2024-03-15',
        accuracy: 82,
        image: viltImg,
    },
    {
        id: 'lxmert',
        name: 'LXMERT',
        type: 'Dual-Stream',
        version: 'v1.0',
        description: 'Cross-modality contextualized representations adapted via dynamic pacing.',
        locked: true,
        image: lxmertImg,
    },
    {
        id: 'oscar',
        name: 'OSCAR',
        type: 'Single-Stream',
        version: 'v1.0',
        description: 'Object-semantics aligned pre-training augmented with concept progression.',
        locked: true,
        image: oscarImg,
    }
];

const filters = ['All Architectures', 'Transformers', 'Dual-Stream'];

const Models = () => {
    const navigate = useNavigate();
    const [selectedModel, setSelectedModel] = useState(null);
    const [drawerVisible, setDrawerVisible] = useState(false);

    const handleCardClick = (model) => {
        if (!model.locked) {
            setSelectedModel(model);
            setDrawerVisible(true);
        }
    };

    return (
        <div className="page-container" style={{ padding: '40px 8%' }}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                <Title level={2} style={{ margin: 0, color: 'var(--text-primary)', fontWeight: 700, letterSpacing: '1px', border: '1px solid var(--glass-border)', display: 'inline-block', padding: '0px 8px' }}>
                    Model Repository
                </Title>
                <Paragraph className="text-secondary" style={{ fontSize: '1rem', marginTop: 10 }}>
                    Explore and configure curriculum-enhanced VQA architectures.
                </Paragraph>

                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', margin: '30px 0', flexWrap: 'wrap' }}>
                    <Text className="text-secondary" style={{ marginRight: 10 }}>Filters:</Text>
                    {filters.map((f, i) => (
                        <div key={f} style={{
                            padding: '6px 16px',
                            background: i === 0 ? 'var(--primary-color)' : 'var(--card-bg)',
                            color: i === 0 ? '#0B1118' : 'var(--text-secondary)',
                            border: i === 0 ? 'none' : '1px solid rgba(0, 229, 255, 0.2)',
                            borderRadius: '20px',
                            cursor: 'pointer',
                            fontSize: '0.9rem',
                            fontWeight: i === 0 ? 600 : 400
                        }}>
                            {f}
                        </div>
                    ))}
                </div>

                <div style={{
                    background: 'rgba(0, 229, 255, 0.05)',
                    border: '1px solid rgba(0, 229, 255, 0.2)',
                    borderRadius: '8px',
                    padding: '24px',
                    marginBottom: '40px'
                }}>
                    <Space align="start">
                        <InfoCircleOutlined style={{ color: 'var(--primary-color)', fontSize: '20px', marginTop: '4px' }} />
                        <div>
                            <Text style={{ color: 'var(--primary-color)', fontWeight: 600, fontSize: '1.1rem', display: 'block', marginBottom: '8px' }}>
                                Note
                            </Text>
                            <Paragraph className="text-secondary" style={{ margin: 0, fontSize: '0.95rem', lineHeight: '1.6' }}>
                                Each model in this repository is a curriculum-enhanced variant of its baseline  to systematically improve compositional reasoning in VQA tasks. Download any enhanced model and benchmark it directly against its flat-trained baseline.
                            </Paragraph>
                        </div>
                    </Space>
                </div>
            </motion.div>

            <Row gutter={[24, 24]}>
                {mockedModels.map((m, i) => (
                    <Col xs={24} sm={12} lg={8} key={m.id}>
                        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.15 }}>
                            <Card
                                className="glass-panel"
                                bordered={false}
                                hoverable={!m.locked}
                                onClick={() => handleCardClick(m)}
                                cover={
                                    <div style={{ height: '140px', overflow: 'hidden', borderTopLeftRadius: '12px', borderTopRightRadius: '12px' }}>
                                        <img alt={m.name} src={m.image} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                    </div>
                                }
                                style={{
                                    height: '380px',
                                    background: 'var(--card-bg)',
                                    border: 'var(--glass-border)',
                                    cursor: m.locked ? 'not-allowed' : 'pointer',
                                    position: 'relative',
                                    opacity: m.locked ? 0.6 : 1,
                                    boxShadow: m.locked ? 'none' : '0 4px 12px rgba(0, 229, 255, 0.05)'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                                    <Tag color={m.locked ? 'default' : 'cyan'} style={{ background: 'transparent', border: m.locked ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid var(--primary-color)', color: m.locked ? 'var(--text-secondary)' : 'var(--primary-color)', borderRadius: '12px', padding: '2px 10px' }}>
                                        {m.type}
                                    </Tag>
                                    <Text className="text-secondary" style={{ fontSize: '0.85rem' }}>{m.version}</Text>
                                </div>

                                <Title level={3} style={{ color: 'var(--text-primary)', marginBottom: '16px', letterSpacing: '0.5px' }}>
                                    {m.name}
                                </Title>
                                <Paragraph className="text-secondary" style={{ fontSize: '1rem', lineHeight: '1.5' }}>
                                    {m.description}
                                </Paragraph>

                                {m.locked && (
                                    <LockOutlined style={{ position: 'absolute', bottom: '24px', right: '24px', fontSize: '24px', color: 'rgba(255, 255, 255, 0.2)' }} />
                                )}
                                {!m.locked && (
                                    <div style={{ position: 'absolute', top: '24px', left: '24px', width: '12px', height: '12px', background: 'var(--primary-color)', borderRadius: '2px', boxShadow: '0 0 8px var(--primary-color)' }} />
                                )}
                            </Card>
                        </motion.div>
                    </Col>
                ))}
            </Row>

            <Drawer
                title={null}
                placement="right"
                onClose={() => setDrawerVisible(false)}
                open={drawerVisible}
                width={450}
                style={{ background: 'var(--bg-color)', color: 'var(--text-primary)', borderLeft: 'var(--glass-border)' }}
                closeIcon={null}
            >
                {selectedModel && (
                    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <div style={{ flex: 1, paddingRight: '10px' }}>
                            <div style={{ marginBottom: '30px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Title level={3} style={{ color: 'var(--primary-color)', margin: 0 }}>{selectedModel.name}</Title>
                                    <Button type="text" onClick={() => setDrawerVisible(false)} style={{ color: 'var(--text-secondary)' }}>✕</Button>
                                </div>
                                <Text className="text-secondary">{selectedModel.type}</Text>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '40px' }}>
                                <div style={{ background: 'var(--card-bg)', padding: '16px 20px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: 'var(--glass-border)' }}>
                                    <Space><HddOutlined style={{ color: 'var(--primary-color)' }} /><Text className="text-secondary">File Size</Text></Space>
                                    <Text style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{selectedModel.size}</Text>
                                </div>
                                <div style={{ background: 'var(--card-bg)', padding: '16px 20px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: 'var(--glass-border)' }}>
                                    <Space><CalendarOutlined style={{ color: 'var(--primary-color)' }} /><Text className="text-secondary">Release Date</Text></Space>
                                    <Text style={{ color: 'var(--primary-color)', fontWeight: 600, border: '1px solid var(--primary-color)', padding: '2px 8px', borderRadius: '4px' }}>{selectedModel.date}</Text>
                                </div>
                                <div style={{ background: 'var(--card-bg)', padding: '16px 20px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: 'var(--glass-border)' }}>
                                    <Space><BranchesOutlined style={{ color: 'var(--primary-color)' }} /><Text className="text-secondary">Version</Text></Space>
                                    <Text style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{selectedModel.version}</Text>
                                </div>
                            </div>

                            <div style={{ marginBottom: '30px' }}>
                                <Text style={{ color: 'var(--primary-color)', fontWeight: 700, letterSpacing: '2px', fontSize: '0.8rem', display: 'block', marginBottom: '16px' }}>METRICS</Text>
                                <div style={{ background: 'var(--card-bg)', padding: '20px', borderRadius: '8px', border: 'var(--glass-border)' }}>
                                    <Space style={{ marginBottom: '12px' }}>
                                        <CheckCircleOutlined style={{ color: 'var(--primary-color)' }} />
                                        <Text style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Overall Accuracy</Text>
                                    </Space>
                                    <div style={{ background: 'rgba(0, 0, 0, 0.4)', height: '40px', borderRadius: '4px', display: 'flex', alignItems: 'center', padding: '0 16px', border: '1px solid rgba(0,229,255,0.1)' }}>
                                        <Text style={{ color: 'var(--primary-color)', fontWeight: 700, fontSize: '1.2rem' }}>{selectedModel.accuracy}%</Text>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.1)', paddingTop: '24px', display: 'flex', gap: '12px' }}>
                            <Button
                                type="primary"
                                style={{ flex: 1, height: '45px', background: 'var(--primary-color)', color: '#0B1118', fontWeight: 600, border: 'none' }}
                                icon={<DownloadOutlined />}
                                onClick={() => {
                                    window.open(getModelDownloadUrl(selectedModel.id), '_blank');
                                }}
                            >
                                Download Weights
                            </Button>
                            <Button
                                type="default"
                                style={{ flex: 1, height: '45px', background: 'transparent', color: 'var(--primary-color)', borderColor: 'var(--primary-color)', fontWeight: 600 }}
                                icon={<ExperimentOutlined />}
                                onClick={() => {
                                    setDrawerVisible(false);
                                    navigate(`/test?model=${selectedModel.id}`);
                                }}
                            >
                                Use in Test
                            </Button>
                            <Button
                                type="text"
                                style={{ height: '45px', color: '#fff', border: '1px solid #555' }}
                                icon={<FileTextOutlined />}
                                onClick={() => {
                                    setDrawerVisible(false);
                                    navigate(`/docs?model=${selectedModel.id}`);
                                }}
                            >
                                Docs
                            </Button>
                        </div>
                    </div>
                )}
            </Drawer>
        </div>
    );
};

export default Models;
