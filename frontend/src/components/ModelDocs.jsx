import React from 'react';
import { Typography, Card, Row, Col, Table, Tag, Space, Button, Divider } from 'antd';
import { ArrowLeftOutlined, DownloadOutlined, RiseOutlined, CheckCircleOutlined, SecurityScanOutlined, FilePdfOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import archImg from '../assets/curriculum_arch.png';

const { Title, Paragraph, Text } = Typography;

const ModelDocs = () => {
    const navigate = useNavigate();

    const overallData = [
        { key: '1', metric: 'Overall Accuracy', baseline: '71.33%', curriculum: '82.65%', delta: '+11.32%' },
    ];

    const tierData = [
        { key: 'L1', tier: 'L1: Attribute & Existence', baseline: '73%', curriculum: '87%', delta: '+14%' },
        { key: 'L2', tier: 'L2: Compare Attribute', baseline: '60%', curriculum: '93%', delta: '+33%' },
        { key: 'L3', tier: 'L3: Counting / Compare Integer', baseline: '40%', curriculum: '53%', delta: '+13%' },
        { key: 'L4', tier: 'L4: Relational', baseline: '80%', curriculum: '93%', delta: '+13%' },
        { key: 'L5', tier: 'L5: Complex Composition', baseline: '33%', curriculum: '67%', delta: '+34%' },
    ];

    const columnStyle = { background: 'transparent', color: 'var(--text-secondary)' };

    return (
        <div className="page-container" style={{ padding: '24px 8%' }}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>

                {/* Header Section */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 30 }}>
                    <div>
                        <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/models')} style={{ color: 'var(--text-secondary)', padding: 0, marginBottom: 16 }}>
                            Back to Models
                        </Button>
                        <Title level={2} style={{ margin: 0, color: 'var(--text-primary)', fontWeight: 700, letterSpacing: '1px' }}>
                            Model Documentation & Report
                        </Title>
                    </div>
                    <Button type="primary" icon={<FilePdfOutlined />} style={{ background: 'var(--primary-color)', color: '#0B1118', fontWeight: 600, border: 'none' }}>
                        Export Report
                    </Button>
                </div>

                <Row gutter={[32, 32]} style={{ marginBottom: 24 }}>
                    <Col xs={24} xl={10}>
                        <Title level={3} style={{ color: 'var(--text-primary)' }}>What is Curriculum Learning?</Title>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.05rem', lineHeight: 1.8 }}>
                            Instead of learning everything at once randomly (which confuses the AI), <strong>Curriculum Learning</strong> trains the model step-by-step, starting from the easiest concepts and gradually introducing harder ones—just like how humans learn in school!
                        </Paragraph>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.05rem', lineHeight: 1.8 }}>
                            Our <strong>Competence-Aware Curriculum Framework</strong> continuously monitors the AI’s "understanding" (competence limit). When the AI masters simple questions, the framework automatically unlocks harder reasoning puzzles without forgetting the basics.
                        </Paragraph>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.05rem', lineHeight: 1.8 }}>
                            Through <strong>Competence Tier Sampling</strong>, the system calculates a dynamic probability distribution across difficulty tiers based on the current competence score. This ensures the model spends sufficient time solidifying foundational knowledge while opportunistically sampling harder examples to push its boundaries.
                        </Paragraph>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.05rem', lineHeight: 1.8 }}>
                            Additionally, <strong>Soft Self-Paced Learning</strong> provides a continuous, graded transition mechanism. Rather than jumping rigidly from Level 1 to Level 2 in discrete chunks, we linearly or exponentially shift the sampling weights, allowing the model to acclimate smoothly to greater complexity and preventing catastrophic forgetting.
                        </Paragraph>

                        <div style={{ marginTop: 30, background: 'rgba(0, 229, 255, 0.05)', padding: 20, borderRadius: 12, border: '1px solid rgba(0,229,255,0.2)' }}>
                            <Title level={4} style={{ color: 'var(--primary-color)', marginTop: 0 }}>How the Dataset is Divided</Title>
                            <Paragraph className="text-secondary" style={{ marginBottom: 0 }}>
                                We structured the VQA questions into 5 tiers of increasing difficulty:
                                <ul style={{ paddingLeft: 20, marginTop: 10 }}>
                                    <li><strong>L1 (Beginner):</strong> Identify simple objects ("Is there a cube?").</li>
                                    <li><strong>L2 (Easy):</strong> Compare basic traits ("Is the sphere red?").</li>
                                    <li><strong>L3 (Intermediate):</strong> Counting and numbers ("How many tiny objects?").</li>
                                    <li><strong>L4 (Hard):</strong> Spatial relations ("What's behind the left cylinder?").</li>
                                    <li><strong>L5 (Expert):</strong> Deep multi-step logic combining the above.</li>
                                </ul>
                            </Paragraph>
                        </div>
                    </Col>
                    
                    <Col xs={24} xl={14}>
                        <Card style={{ background: 'var(--card-bg)', border: 'var(--glass-border)', borderRadius: 12, marginBottom: 24 }}>
                            <Title level={4} style={{ color: 'var(--text-primary)', marginBottom: 20 }}>Baseline vs Curriculum Performance</Title>
                            <Table
                                dataSource={overallData}
                                pagination={false}
                                className="dark-table"
                                columns={[
                                    { title: 'Metric', dataIndex: 'metric', key: 'metric', align: 'left' },
                                    { title: 'Baseline ViLT', dataIndex: 'baseline', key: 'baseline', align: 'center' },
                                    { title: 'Curriculum ViLT (Ours)', dataIndex: 'curriculum', key: 'curriculum', align: 'center', render: t => <Text style={{ color: 'var(--primary-color)', fontWeight: 600 }}>{t}</Text> },
                                    { title: 'Gain', dataIndex: 'delta', key: 'delta', align: 'center', render: t => <Tag color="success">{t}</Tag> },
                                ]}
                            />
                        </Card>
                        
                        <Card style={{ background: '#fff', border: 'none', borderRadius: 12, overflow: 'hidden', padding: 10, marginBottom: 24 }}>
                            <Title level={4} style={{ color: '#333', marginTop: 0 }}>Curriculum Framework Architecture</Title>
                            <img src={archImg} alt="Curriculum Architecture" style={{ width: '100%', height: 'auto', borderRadius: 8 }} />
                        </Card>
                        
                        <Card style={{ background: 'var(--card-bg)', border: 'var(--glass-border)', borderRadius: 12 }}>
                            <Title level={4} style={{ color: 'var(--text-primary)', marginBottom: 20 }}>Tier-Wise Accuracy Metrics</Title>
                            <Table
                                dataSource={tierData}
                                pagination={false}
                                className="dark-table"
                                columns={[
                                    { title: 'Difficulty Tier', dataIndex: 'tier', key: 'tier', align: 'left' },
                                    { title: 'Baseline', dataIndex: 'baseline', key: 'baseline', align: 'center' },
                                    { title: 'Curriculum (Ours)', dataIndex: 'curriculum', key: 'curriculum', align: 'center', render: t => <Text style={{ color: 'var(--primary-color)', fontWeight: 600 }}>{t}</Text> },
                                    { title: 'Improvement', dataIndex: 'delta', key: 'delta', align: 'center', render: t => <Text style={{ color: '#52c41a', fontWeight: 600 }}>{t}</Text> },
                                ]}
                            />
                        </Card>
                    </Col>
                </Row>

                {/* CSS to handle the dark table inner colors override smoothly */}
                <style dangerouslySetInnerHTML={{
                    __html: `
                    .dark-table .ant-table { background: transparent !important; color: var(--text-secondary); }
                    .dark-table .ant-table-thead > tr > th { background: rgba(0,0,0,0.3) !important; color: var(--text-primary); border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
                    .dark-table .ant-table-tbody > tr > td { border-bottom: 1px solid rgba(255,255,255,0.05) !important; color: var(--text-secondary) !important; }
                    .dark-table .ant-table-tbody > tr:hover > td { background: rgba(0, 229, 255, 0.05) !important; }
                    .dark-table .ant-empty-description { color: var(--text-secondary); }
                `}} />

            </motion.div>
        </div>
    );
};

export default ModelDocs;
