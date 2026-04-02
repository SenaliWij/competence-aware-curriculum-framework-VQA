import { useState, useEffect } from 'react';
import { Row, Col, Input, Button, Card, Typography, Upload, Progress, List, Tag, Select, Alert } from 'antd';
import { InboxOutlined, SendOutlined, LockOutlined, WarningOutlined } from '@ant-design/icons';
import { useSearchParams } from 'react-router-dom';
import { predictVQA, getModels } from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Dragger } = Upload;

// Model IDs that the backend actually supports for inference.
const AVAILABLE_MODEL_IDS = ['vilt_curriculum', 'vilt_baseline'];

const VQAPlayground = () => {
    const [searchParams] = useSearchParams();
    const [selectedModel, setSelectedModel] = useState(searchParams.get('model') || 'vilt_curriculum');
    const [models, setModels] = useState([]);
    const [question, setQuestion] = useState('');
    const [image, setImage] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [preview, setPreview] = useState('');
    const [error, setError] = useState('');

    useEffect(() => {
        getModels().then(setModels).catch(console.error);
    }, []);

    // If the backend returns other models (future work), we treat them as locked here.
    const isLockedModel = (modelId) => !AVAILABLE_MODEL_IDS.includes(modelId);

    const handlePredict = async () => {
        setError('');

        // Locked model guard
        if (isLockedModel(selectedModel)) {
            setError('This model is not available yet. Please select an available model.');
            return;
        }

        // Validation: require both image and question
        if (!image && !question.trim()) {
            setError('Please upload an image and enter a question before running inference.');
            return;
        }
        if (!image) {
            setError('Please upload an image before running inference.');
            return;
        }
        if (!question.trim()) {
            setError('Please enter a question before running inference.');
            return;
        }

        setLoading(true);
        try {
            const res = await predictVQA(question, image, selectedModel);
            setResult(res);
        } catch (err) {
            console.error(err);
            setError('Inference failed. Make sure the backend is running and try again.');
        }
        setLoading(false);
    };

    const handleUpload = ({ file }) => {
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setError('');
        return false;
    };

    const locked = isLockedModel(selectedModel);

    return (
        <div style={{ padding: '40px 4%', minHeight: '100vh' }}>
            <Title level={2} style={{ marginBottom: 8, fontSize: '2.2rem' }}>
                VQA Testing <span className="text-accent">Interface</span>
            </Title>
            <Paragraph className="text-secondary" style={{ fontSize: '1.15rem', marginBottom: 30 }}>
                Select a model, upload an image, ask a question, and see the answer.
            </Paragraph>

            {/* Model selector */}
            <div style={{ marginBottom: 24 }}>
                <Text style={{ display: 'block', marginBottom: 10, fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
                    Active Model
                </Text>
                <Select
                    value={selectedModel}
                    onChange={(val) => { setSelectedModel(val); setResult(null); setError(''); }}
                    style={{ width: 360, fontSize: '1.05rem' }}
                    options={models.map(m => ({ value: m.id, label: m.name }))}
                    dropdownStyle={{ background: '#0d1f2d', color: '#fff' }}
                />
            </div>

            {/* Locked model warning */}
            {locked && (
                <Alert
                    message={
                        <span style={{ fontSize: '1.05rem', fontWeight: 600 }}>
                            <LockOutlined style={{ marginRight: 8 }} />
                            Not Available Yet
                        </span>
                    }
                    description={
                        <span style={{ fontSize: '1rem' }}>
                            The selected model is not yet available for inference. Please select a different model.
                        </span>
                    }
                    type="warning"
                    showIcon={false}
                    style={{ marginBottom: 24, background: 'rgba(255, 170, 0, 0.08)', border: '1px solid rgba(255,170,0,0.3)', borderRadius: 8 }}
                />
            )}

            {/* Validation / API error */}
            {error && !locked && (
                <Alert
                    message={
                        <span style={{ fontSize: '1rem' }}>
                            <WarningOutlined style={{ marginRight: 8 }} />
                            {error}
                        </span>
                    }
                    type="error"
                    closable
                    onClose={() => setError('')}
                    style={{ marginBottom: 24, background: 'rgba(255, 50, 50, 0.08)', border: '1px solid rgba(255,50,50,0.3)', borderRadius: 8 }}
                />
            )}

            <Row gutter={[28, 28]}>
                {/* Left: Input panel */}
                <Col xs={24} lg={12}>
                    <Card className="glass-panel" style={{ height: '100%' }}>
                        <Title level={4} style={{ fontSize: '1.25rem', marginBottom: 16 }}>
                            <InboxOutlined style={{ marginRight: 8 }} />
                            Image Input
                        </Title>
                        <Dragger
                            className="upload-dragger"
                            beforeUpload={() => false}
                            onChange={handleUpload}
                            showUploadList={false}
                            disabled={locked}
                        >
                            {preview ? (
                                <img
                                    src={preview}
                                    alt="preview"
                                    style={{ maxHeight: '260px', maxWidth: '100%', borderRadius: '8px' }}
                                />
                            ) : (
                                <div style={{ padding: '32px 20px' }}>
                                    <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
                                        Click or drag file to this area to upload
                                    </p>
                                </div>
                            )}
                        </Dragger>

                        <div style={{ marginTop: 32 }}>
                            <Title level={4} style={{ fontSize: '1.25rem', marginBottom: 12 }}>Your Question</Title>
                            <TextArea
                                rows={4}
                                style={{ fontSize: '1.05rem', lineHeight: '1.6' }}
                                value={question}
                                onChange={e => { setQuestion(e.target.value); setError(''); }}
                                placeholder="e.g., Is the cat sitting on a striped rug?"
                                className="input-dark"
                                disabled={locked}
                            />
                            <Button
                                type="primary"
                                className="neon-button"
                                icon={<SendOutlined />}
                                onClick={handlePredict}
                                loading={loading}
                                disabled={locked}
                                style={{ marginTop: 16, width: '100%', height: 48, fontSize: '1.1rem' }}
                            >
                                Get Answer
                            </Button>
                        </div>
                    </Card>
                </Col>

                {/* Right: Output panel */}
                <Col xs={24} lg={12}>
                    {result && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            <Card className="glass-panel">
                                <Title level={4} style={{ fontSize: '1.25rem', marginBottom: 12 }}>Model Answer</Title>
                                <Title level={2} style={{ color: '#00e5ff', margin: '10px 0', fontSize: '2rem' }}>
                                    {result.answer}.
                                </Title>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
                                    <Text style={{ fontSize: '1.1rem' }}>
                                        Confidence: {Math.round(result.confidence * 100)}%
                                    </Text>
                                    <Tag color="cyan" style={{ fontSize: '0.95rem', padding: '3px 10px' }}>
                                        {models.find(m => m.id === selectedModel)?.name || selectedModel}
                                    </Tag>
                                </div>
                            </Card>

                            <Card className="glass-panel">
                                <Title level={4} style={{ fontSize: '1.25rem', marginBottom: 16 }}>Candidate Answers</Title>
                                <List
                                    dataSource={result.candidate_answers}
                                    renderItem={item => (
                                        <List.Item style={{ borderBottom: '1px solid rgba(255,255,255,0.07)', paddingBottom: 12 }}>
                                            <div style={{ width: '100%' }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                                    <Text style={{ fontSize: '1.1rem', fontWeight: 500 }}>{item.text}</Text>
                                                    <Text style={{ fontSize: '1.05rem', color: 'var(--primary-color)', marginLeft: 16 }}>
                                                        {Math.round(item.confidence * 100)}%
                                                    </Text>
                                                </div>
                                                <Progress
                                                    percent={Math.round(item.confidence * 100)}
                                                    showInfo={false}
                                                    strokeColor="#00e5ff"
                                                    trailColor="rgba(255,255,255,0.07)"
                                                />
                                            </div>
                                        </List.Item>
                                    )}
                                />
                            </Card>
                        </div>
                    )}
                    {!result && (
                        <div style={{
                            height: '100%',
                            minHeight: 340,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            border: '1px dashed rgba(255,255,255,0.12)',
                            borderRadius: 12,
                        }}>
                            <Text style={{ color: '#555', fontSize: '1.05rem' }}>
                                Model output will appear here
                            </Text>
                        </div>
                    )}
                </Col>
            </Row>
        </div>
    );
};

export default VQAPlayground;
