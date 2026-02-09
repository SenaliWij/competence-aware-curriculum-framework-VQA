import { useState } from 'react';
import { Row, Col, Card, Steps, Button, Typography, Form, InputNumber, Switch } from 'antd';
import { CheckCircleFilled } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { startTraining } from '../services/api';

const { Title, Text } = Typography;

const ModelConfig = () => {
    const navigate = useNavigate();
    const [currentStep, setCurrentStep] = useState(0);
    const [selectedModel, setSelectedModel] = useState(null);
    const [config, setConfig] = useState({
        batch_size: 32,
        epochs: 15,
        learning_rate: 0.001,
        use_curriculum: true
    });

    const models = [
        { id: 'vilt', name: 'ViLT Model', desc: 'Optimized for visual-language tasks with efficient training.', color: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%)' },
        { id: 'lxmert', name: 'LXMERT Model', desc: 'Combines visual and textual information for enhanced reasoning.', color: 'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)' },
        { id: 'albef', name: 'ALBEF Model', desc: 'Aligns image and text features for robust understanding.', color: 'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)' },
    ];

    const handleStartTraining = async () => {
        try {
            await startTraining({
                model_name: selectedModel,
                ...config
            });
            navigate('/training');
        } catch (error) {
            console.error("Failed to start training", error);
        }
    };

    return (
        <div className="page-container h-full">
            <Title level={2} className="mb-40">Configuration Wizard</Title>

            <Steps
                current={currentStep}
                className="mb-40"
                style={{ marginBottom: '60px' }}
                items={[
                    { title: 'Select Model' },
                    { title: 'Set Parameters' },
                    { title: 'Review & Start' },
                ]}
            />

            <div style={{ minHeight: '400px' }}>
                {currentStep === 0 && (
                    <Row gutter={[24, 24]}>
                        {models.map((model) => (
                            <Col xs={24} md={8} key={model.id}>
                                <Card
                                    className={`glass-panel model-card-wrapper ${selectedModel === model.id ? 'model-card-active' : 'model-card-inactive'}`}
                                    hoverable
                                    onClick={() => setSelectedModel(model.id)}
                                    cover={
                                        <div className="model-color-header" style={{ background: model.color }}>
                                            {selectedModel === model.id && <CheckCircleFilled className="model-check-icon" />}
                                        </div>
                                    }
                                >
                                    <Card.Meta title={model.name} description={<span className="text-secondary">{model.desc}</span>} />
                                </Card>
                            </Col>
                        ))}
                    </Row>
                )}

                {currentStep === 1 && (
                    <div className="glass-panel config-form-panel">
                        <Form layout="vertical">
                            <Form.Item label="Batch Size">
                                <InputNumber className="w-full" value={config.batch_size} onChange={v => setConfig({ ...config, batch_size: v })} />
                            </Form.Item>
                            <Form.Item label="Epochs">
                                <InputNumber className="w-full" value={config.epochs} onChange={v => setConfig({ ...config, epochs: v })} />
                            </Form.Item>
                            <Form.Item label="Curriculum Learning">
                                <Switch checked={config.use_curriculum} onChange={v => setConfig({ ...config, use_curriculum: v })} />
                            </Form.Item>
                        </Form>
                    </div>
                )}

                {currentStep === 2 && (
                    <div className="glass-panel review-panel">
                        <Title level={3}>Ready to Train</Title>
                        <Text className="review-model-text">Model: <span className="text-accent">{selectedModel}</span></Text>
                        <Text className="text-secondary">Configuration: {config.epochs} epochs, Batch size {config.batch_size}</Text>
                    </div>
                )}
            </div>

            <div className="flex-end-gap mt-40">
                {currentStep > 0 && <Button onClick={() => setCurrentStep(currentStep - 1)}>Back</Button>}
                {currentStep < 2 && <Button type="primary" disabled={currentStep === 0 && !selectedModel} onClick={() => setCurrentStep(currentStep + 1)}>Next</Button>}
                {currentStep === 2 && <Button type="primary" className="neon-button" onClick={handleStartTraining}>Start Training</Button>}
            </div>
        </div>
    );
};

export default ModelConfig;
