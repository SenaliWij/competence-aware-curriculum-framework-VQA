import { Typography, Card, Row, Col, Table, Tag, Button } from 'antd';
import { ArrowLeftOutlined, TrophyOutlined, WarningOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

import archImg from '../assets/curriculum_arch.png';
import advantageImg from '../assets/curriculum_advantage_table.png';
import failureImg from '../assets/complete_failure_table.png';

const { Title, Paragraph, Text } = Typography;

const overallData = [
    { key: '1', metric: 'Overall Accuracy', baseline: '71.33%', curriculum: '82.65%', delta: '+11.32%' },
];

const tierData = [
    { key: '1', tier: 'Tier 1', reasoning: 'Attribute & Existence', baseline: '82.08%', curriculum: '91.23%' },
    { key: '2', tier: 'Tier 2', reasoning: 'Compare Attribute', baseline: '63.44%', curriculum: '97.01%' },
    { key: '3', tier: 'Tier 3', reasoning: 'Counting / Compare Int.', baseline: '67.71%', curriculum: '75.94%' },
    { key: '4', tier: 'Tier 4', reasoning: 'Relational', baseline: '86.45%', curriculum: '85.16%' },
    { key: '5', tier: 'Tier 5', reasoning: 'Complex Composition', baseline: '58.91%', curriculum: '69.40%' },
];

const chartData = [
    { tier: 'T1', 'B-ViLT': 82.08, 'CL-ViLT': 91.23 },
    { tier: 'T2', 'B-ViLT': 63.44, 'CL-ViLT': 97.01 },
    { tier: 'T3', 'B-ViLT': 67.71, 'CL-ViLT': 75.94 },
    { tier: 'T4', 'B-ViLT': 86.45, 'CL-ViLT': 85.16 },
    { tier: 'T5', 'B-ViLT': 58.91, 'CL-ViLT': 69.40 },
];

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: '#0d1a2d', border: '1px solid rgba(0,229,255,0.3)',
            borderRadius: 8, padding: '10px 14px',
        }}>
            <p style={{ color: '#fff', margin: '0 0 6px', fontWeight: 600 }}>{`Tier ${label}`}</p>
            {payload.map(p => (
                <p key={p.name} style={{ color: p.color, margin: '2px 0', fontSize: '0.9rem' }}>
                    {`${p.name}: ${p.value.toFixed(2)}%`}
                </p>
            ))}
        </div>
    );
};

const ModelDocs = () => {
    const navigate = useNavigate();

    const tierColumns = [
        { title: 'Tier', dataIndex: 'tier', key: 'tier', width: 70 },
        { title: 'Reasoning Type', dataIndex: 'reasoning', key: 'reasoning' },
        {
            title: 'B-ViLT', dataIndex: 'baseline', key: 'baseline', align: 'center',
            render: t => <Text style={{ color: '#ff9800', fontWeight: 600 }}>{t}</Text>,
        },
        {
            title: 'CL-ViLT (Ours)', dataIndex: 'curriculum', key: 'curriculum', align: 'center',
            render: t => <Text style={{ color: 'var(--primary-color)', fontWeight: 700 }}>{t}</Text>,
        },
        {
            title: 'Δ Gain', key: 'gain', align: 'center',
            render: (_, row) => {
                const b = parseFloat(row.baseline);
                const c = parseFloat(row.curriculum);
                const diff = (c - b).toFixed(2);
                const positive = c >= b;
                return (
                    <Tag color={positive ? 'success' : 'error'} style={{ fontWeight: 600 }}>
                        {positive ? '+' : ''}{diff}%
                    </Tag>
                );
            },
        },
    ];

    return (
        <div className="page-container" style={{ padding: '24px 6%' }}>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>

                {/* Header */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 28 }}>
                    <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/models')}
                        style={{ color: 'var(--text-secondary)', padding: 0 }}>
                        Back to Models
                    </Button>
                </div>
                <Title level={2} style={{ margin: '0 0 6px', color: 'var(--text-primary)', fontWeight: 700, letterSpacing: '1px' }}>
                    Training Stratergy Documentation
                </Title>
                <Paragraph className="text-secondary" style={{ fontSize: '1.05rem', marginBottom: 32 }}>
                    Competence-aware curriculum learning for Visual Question Answering: methodology, architecture, & results.
                </Paragraph>
                {/* Training stratergy architecture explanation*/}
                <Row gutter={[28, 28]} style={{ marginBottom: 28 }}>
                    <Col xs={24} xl={11}>
                        <Card style={{ background: 'var(--card-bg)', border: 'var(--glass-border)', borderRadius: 12, height: '100%' }}>
                            <Title level={4} style={{ color: 'var(--text-primary)', marginTop: 0 }}>What is Curriculum Learning?</Title>
                            <Paragraph className="text-secondary" style={{ fontSize: '1rem', lineHeight: 1.8 }}>
                                Instead of learning everything at once randomly, <strong>Curriculum Learning</strong> trains the model
                                step-by-step - from easiest concepts to harder ones, just like how humans learn in school.
                            </Paragraph>
                            <Paragraph className="text-secondary" style={{ fontSize: '1rem', lineHeight: 1.8 }}>
                                Our <strong>Competence-Aware Training Stratergy</strong> continuously monitors the model's competence.
                                The Competence Tracker monitors the model's learning by smoothing out noisy batch signals (entropy and loss) using an Exponential Moving Average, producing a single Competence Score (C) between 0 (beginner) and 1 (mastered). The Tier Sampler uses that score to probabilistically select a difficulty tier , when C is low, easy tiers dominate; as C rises, harder tiers unlock gradually. The Soft Self-Paced Sampler then fine-tunes within the chosen tier by weighting individual samples, favouring easy examples when the model is still learning and harder ones as competence grows.
                            </Paragraph>
                            <div style={{ background: 'rgba(0,229,255,0.05)', padding: 16, borderRadius: 10, border: '1px solid rgba(0,229,255,0.2)', marginTop: 8 }}>
                                <Title level={5} style={{ color: 'var(--primary-color)', marginTop: 0 }}>5 Difficulty Tiers</Title>
                                <div style={{ display: 'grid', gap: 6 }}>
                                    {[
                                        ['L1', 'Attribute & Existence'],
                                        ['L2', 'Compare Attribute'],
                                        ['L3', 'Counting & Compare Integer'],
                                        ['L4', 'Relational'],
                                        ['L5', 'Complex Composition'],
                                    ].map(([l, type, ex]) => (
                                        <div key={l} style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
                                            <Tag style={{ background: 'rgba(0,229,255,0.1)', border: '1px solid rgba(0,229,255,0.3)', color: 'var(--primary-color)', minWidth: 28, textAlign: 'center' }}>{l}</Tag>
                                            <Text style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.9rem' }}>{type}</Text>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </Card>
                    </Col>
                    {/* Training stratergy architecture image */}
                    <Col xs={24} xl={13}>
                        <Card style={{ background: 'var(--card-bg)', border: 'none', borderRadius: 12, overflow: 'hidden', height: '100%', display: 'flex', alignItems: 'center' }}>
                            <img src={archImg} alt="Curriculum Architecture" style={{ width: '100%', height: 'auto', display: 'block' }} />
                        </Card>
                    </Col>
                </Row>

                {/* Overall accuracy + Tier table + Bar chart */}
                <Title level={3} style={{ color: 'var(--text-primary)', marginBottom: 16 }}>Performance Results</Title>
                <Row gutter={[28, 28]} style={{ marginBottom: 28 }}>

                    {/* Overall and  Tier-wise accuracy table */}
                    <Col xs={24} lg={13}>
                        <Card style={{ background: 'var(--card-bg)', border: 'var(--glass-border)', borderRadius: 12, marginBottom: 16 }}>
                            <Title level={5} style={{ color: 'var(--text-primary)', marginTop: 0, marginBottom: 14 }}>Overall Accuracy</Title>
                            <Table
                                dataSource={overallData} pagination={false} className="dark-table"
                                columns={[
                                    { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                                    { title: 'B-ViLT', dataIndex: 'baseline', key: 'baseline', align: 'center', render: t => <Text style={{ color: '#ff9800', fontWeight: 600 }}>{t}</Text> },
                                    { title: 'CL-ViLT (Ours)', dataIndex: 'curriculum', key: 'curriculum', align: 'center', render: t => <Text style={{ color: 'var(--primary-color)', fontWeight: 700 }}>{t}</Text> },
                                    { title: 'Gain', dataIndex: 'delta', key: 'delta', align: 'center', render: t => <Tag color="success">{t}</Tag> },
                                ]}
                            />
                        </Card>
                        <Card style={{ background: 'var(--card-bg)', border: 'var(--glass-border)', borderRadius: 12 }}>
                            <Title level={5} style={{ color: 'var(--text-primary)', marginTop: 0, marginBottom: 14 }}>Tier-Wise Accuracy</Title>
                            <Table
                                dataSource={tierData} pagination={false} className="dark-table"
                                columns={tierColumns}
                                size="small"
                            />
                        </Card>
                    </Col>

                    {/* Bar chart */}
                    <Col xs={24} lg={11}>
                        <Card style={{ background: 'var(--card-bg)', border: 'var(--glass-border)', borderRadius: 12, height: '100%' }}>
                            <Title level={5} style={{ color: 'var(--text-primary)', marginTop: 0, marginBottom: 4 }}>
                                Tier Accuracy Comparison
                            </Title>
                            <Paragraph className="text-secondary" style={{ fontSize: '0.9rem', marginBottom: 20 }}>
                                <span style={{ color: '#ff9800', fontWeight: 600 }}>■ B-ViLT</span> (Baseline) vs{' '}
                                <span style={{ color: '#00e5ff', fontWeight: 600 }}>■ CL-ViLT</span> (Curriculum)
                            </Paragraph>
                            <ResponsiveContainer width="100%" height={280}>
                                <BarChart data={chartData} margin={{ top: 4, right: 8, left: -8, bottom: 4 }} barCategoryGap="28%">
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                                    <XAxis dataKey="tier" tick={{ fill: '#8899aa', fontSize: 13 }} axisLine={false} tickLine={false} />
                                    <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#8899aa', fontSize: 11 }} axisLine={false} tickLine={false} />
                                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                                    <Bar dataKey="B-ViLT" fill="#ff9800" radius={[4, 4, 0, 0]} maxBarSize={36} />
                                    <Bar dataKey="CL-ViLT" fill="#00e5ff" radius={[4, 4, 0, 0]} maxBarSize={36} />
                                </BarChart>
                            </ResponsiveContainer>
                            <Paragraph className="text-secondary" style={{ fontSize: '0.82rem', marginTop: 8, marginBottom: 0 }}>
                                * Tier 4 is the only tier where baseline slightly outperforms curriculum (86.45% vs 85.16%),
                                suggesting spatial-relational questions may benefit from flat-training exposure.
                            </Paragraph>
                        </Card>
                    </Col>
                </Row>

                {/* Case Studies */}
                <Row gutter={[28, 28]} style={{ marginBottom: 28 }}>

                    {/* Curriculum Advantage Cases */}
                    <Col xs={24} xl={12}>
                        <div style={{ marginBottom: 28 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                                <TrophyOutlined style={{ color: '#00e5ff', fontSize: 20 }} />
                                <Title level={3} style={{ margin: 0, color: 'var(--text-primary)' }}>Curriculum Advantage Cases</Title>
                            </div>
                            <Paragraph className="text-secondary" style={{ fontSize: '1rem', marginBottom: 16 }}>
                                Cases where <strong style={{ color: '#00e5ff' }}>CL-ViLT answers correctly</strong> but <strong style={{ color: '#ff9800' }}>B-ViLT fails</strong>.
                            </Paragraph>

                            <Card style={{ background: 'transparent', border: 'none', borderRadius: 12, overflow: 'hidden', padding: 0, display: 'flex', justifyContent: 'center' }}>
                                <img
                                    src={advantageImg}
                                    alt="Curriculum Advantage Cases"
                                    style={{ maxWidth: '100%', height: 'auto', display: 'block', borderRadius: 6 }}
                                />
                            </Card>
                        </div>
                    </Col>

                    {/* Complete Failure Cases */}
                    <Col xs={24} xl={12}>
                        <div style={{ marginBottom: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                                <WarningOutlined style={{ color: '#ff9800', fontSize: 20 }} />
                                <Title level={3} style={{ margin: 0, color: 'var(--text-primary)' }}>Complete Failure Cases</Title>
                            </div>
                            <Paragraph className="text-secondary" style={{ fontSize: '1rem', marginBottom: 16 }}>
                                Cases where <strong style={{ color: '#ff9800' }}>all models fail</strong> highlighting the limits of current architectures.
                            </Paragraph>

                            <Card style={{ background: 'transparent', border: 'none', borderRadius: 12, overflow: 'hidden', padding: 0, display: 'flex', justifyContent: 'center' }}>
                                <img
                                    src={failureImg}
                                    alt="Complete Failure Cases"
                                    style={{ maxWidth: '53%', height: 'auto', display: 'block', borderRadius: 6 }}
                                />
                            </Card>
                        </div>
                    </Col>
                </Row>

            </motion.div>

            {/* Dark table styles */}
            <style dangerouslySetInnerHTML={{
                __html: `
                .dark-table .ant-table { background: transparent !important; }
                .dark-table .ant-table-thead > tr > th { background: rgba(0,0,0,0.3) !important; color: var(--text-primary); border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
                .dark-table .ant-table-tbody > tr > td { border-bottom: 1px solid rgba(255,255,255,0.05) !important; color: var(--text-secondary) !important; }
                .dark-table .ant-table-tbody > tr:hover > td { background: rgba(0,229,255,0.04) !important; }
                .dark-table .ant-empty-description { color: var(--text-secondary); }
            ` }} />
        </div>
    );
};

export default ModelDocs;
