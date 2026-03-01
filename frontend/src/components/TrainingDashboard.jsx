import { useState, useEffect } from 'react';
import { Row, Col, Card, Typography, Button, Spin, Statistic } from 'antd';
import { PauseCircleOutlined, StopOutlined, RiseOutlined } from '@ant-design/icons';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { getTrainingStatus, stopTraining } from '../services/api';

const { Title, Text } = Typography;

const TrainingDashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const status = await getTrainingStatus();
                setData(status);
                setLoading(false);
            } catch (err) {
                console.error(err);
            }
        };

        fetchData(); // Initial fetch
        const interval = setInterval(fetchData, 2000); // Polling every 2s

        return () => clearInterval(interval);
    }, []);

    const handleStop = async () => {
        await stopTraining();
    };

    if (!data) return <div className="text-center" style={{ padding: '50px' }}><Spin size="large" /></div>;

    const progressData = [
        { name: 'Completed', value: data.metrics.accuracy },
        { name: 'Remaining', value: 100 - data.metrics.accuracy },
    ];
    const COLORS = ['#00e5ff', '#333'];

    return (
    <div className="dashboard-container">
        <Row gutter={40} align="top">
        
        {/* LEFT: Training Panel */}
        <Col span={14}>
            <div className="glass-panel dashboard-header-panel">
            <Title level={2}>
                {data.is_training ? 'Training ' : 'Training '}
            </Title>

            <Text className="text-secondary" style={{ display: 'block', marginBottom: '40px' }}>
                {data.status_message}
            </Text>

            {/* Accuracy Ring */}
            <div className="chart-container-wrapper">
                <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                    <Pie
                    data={progressData}
                    innerRadius={80}
                    outerRadius={100}
                    startAngle={90}
                    endAngle={-270}
                    dataKey="value"
                    stroke="none"
                    >
                    {progressData.map((_, index) => (
                        <Cell key={index} fill={COLORS[index]} />
                    ))}
                    </Pie>
                </PieChart>
                </ResponsiveContainer>

                <div className="chart-center-label">
                <Title level={3} style={{ margin: 0, color: '#00e5ff' }}>
                    {data.metrics.accuracy}%
                </Title>
                <Text style={{ fontSize: '12px' }}>Progress</Text>
                </div>
            </div>

            {/* Core stats */}
            <Row className="mt-40" gutter={24}>
                <Col span={12}>
                {/* <Statistic
                    title="Global Step"
                    value={data.metrics.global_step}
                    valueStyle={{ color: '#fff' }}
                /> */}
                </Col>
                <Col span={12}>
                {/* <Statistic
                    title="Loss"
                    value={data.metrics.loss}
                    precision={4}
                    valueStyle={{ color: '#ff4d4f' }}
                /> */}
                </Col>
            </Row>
            </div>

            {/* Buttons */}
            <div className="flex-center-gap mt-24">
            <Button size="large" icon={<PauseCircleOutlined />}>
                Pause Training
            </Button>
            <Button
                size="large"
                danger
                type="primary"
                icon={<StopOutlined />}
                onClick={handleStop}
            >
                Stop Training
            </Button>
            </div>
        </Col>

        {/* RIGHT: Metrics Sidebar */}
        <Col span={10}>
            <div className="metrics-sidebar">

                <Card className="glass-panel mb-24">
                    <Statistic
                        title="Tier 1 Accuracy"
                        value={82.9}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: '#00e5ff' }}
                    />
                </Card>

                <Card className="glass-panel mb-24">
                    <Statistic
                        title="Tier 2 Accuracy"
                        value={data.metrics.tier_accuracy?.tier2}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: '#00e5ff' }}
                    />
                </Card>

                <Card className="glass-panel mb-24">
                    <Statistic
                        title="Tier 3 Accuracy"
                        value={data.metrics.tier_accuracy?.tier3}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: '#00e5ff' }}
                    />
                </Card>

                <Card className="glass-panel mb-24">
                    <Statistic
                        title="Tier 4 Accuracy"
                        value={data.metrics.tier_accuracy?.tier4}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: '#00e5ff' }}
                    />
                </Card>

                <Card className="glass-panel">
                    <Statistic
                        title="Tier 5 Accuracy"
                        value={data.metrics.tier_accuracy?.tier5}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: '#00e5ff' }}
                    />
                </Card>

            </div>

        </Col>

        </Row>
    </div>
    );

};

export default TrainingDashboard;
