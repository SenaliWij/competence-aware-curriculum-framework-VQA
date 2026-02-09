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
            <div className="glass-panel dashboard-header-panel">
                <Title level={2}>{data.is_training ? 'Training Live' : 'Training Completed'}</Title>
                <Text className="text-secondary" style={{ display: 'block', marginBottom: '40px' }}>{data.status_message}</Text>

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
                                {progressData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="chart-center-label">
                        <Title level={3} style={{ margin: 0, color: '#00e5ff' }}>{data.metrics.accuracy}%</Title>
                        <Text style={{ fontSize: '12px' }}>Accuracy</Text>
                    </div>
                </div>

                <Row className="mt-40" gutter={24}>
                    <Col span={8}>
                        <Statistic title="Epoch" value={data.metrics.epoch} valueStyle={{ color: '#fff' }} />
                    </Col>
                    <Col span={8}>
                        <Statistic title="Global Step" value={data.metrics.global_step} valueStyle={{ color: '#fff' }} />
                    </Col>
                    <Col span={8}>
                        <Statistic title="Loss" value={data.metrics.loss} precision={4} valueStyle={{ color: '#ff4d4f' }} />
                    </Col>
                </Row>
            </div>

            <div className="flex-center-gap">
                <Button size="large" icon={<PauseCircleOutlined />}>Pause Training</Button>
                <Button size="large" danger type="primary" icon={<StopOutlined />} onClick={handleStop}>Stop Training</Button>
            </div>

            <Row gutter={[24, 24]} className="mt-40">
                <Col span={8}>
                    <Card className="glass-panel">
                        <Statistic title="Compositional Reasoning" value={88.5} precision={1} suffix="%" valueStyle={{ color: '#00e5ff' }} prefix={<RiseOutlined />} />
                        <Text type="secondary">2.1% higher than previous epoch</Text>
                    </Card>
                </Col>
                <Col span={8}>
                    <Card className="glass-panel">
                        <Statistic title="Overall VQA Accuracy" value={data.metrics.accuracy} precision={1} suffix="%" valueStyle={{ color: '#00e5ff' }} />
                        <Text type="secondary">Trending upwards</Text>
                    </Card>
                </Col>
                <Col span={8}>
                    <Card className="glass-panel">
                        <Statistic title="Model Robustness" value={0.92} precision={2} valueStyle={{ color: '#00e5ff' }} />
                        <Text type="secondary">Stable across tiers</Text>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default TrainingDashboard;
