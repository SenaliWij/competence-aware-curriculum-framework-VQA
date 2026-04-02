import { useState, useEffect } from 'react';
import { Row, Col, Input, Button, Card, Typography, Upload, Progress, List, Tag, Select } from 'antd';
import { InboxOutlined, SendOutlined } from '@ant-design/icons';
import { useSearchParams } from 'react-router-dom';
import { predictVQA, getModels } from '../services/api';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Dragger } = Upload;

const VQAPlayground = () => {
    const [searchParams] = useSearchParams();
    const [selectedModel, setSelectedModel] = useState(searchParams.get('model') || 'vilt_curriculum');
    const [models, setModels] = useState([]);
    const [question, setQuestion] = useState('');
    const [image, setImage] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [preview, setPreview] = useState('');

    useEffect(() => {
        getModels().then(setModels).catch(console.error);
    }, []);

    const handlePredict = async () => {
        if (!question) return;
        setLoading(true);
        try {
            const res = await predictVQA(question, image, selectedModel);
            setResult(res);
        } catch (err) {
            console.error(err);
        }
        setLoading(false);
    };

    const handleUpload = ({ file }) => {
        setImage(file);
        setPreview(URL.createObjectURL(file));
        return false; // Prevent auto upload
    };

    return (
        <div className="playground-container">
            <Title level={2} className="mb-20">VQA Testing <span className="text-accent">Interface</span></Title>
            <Text className="text-secondary mb-40" style={{ display: 'block' }}>
                Select a model, upload an image, ask a question, and see the answer.
            </Text>

            {/* Model selector */}
            <div style={{ marginBottom: 30 }}>
                <Text style={{ display: 'block', marginBottom: 8 }}>Active Model</Text>
                <Select
                    value={selectedModel}
                    onChange={setSelectedModel}
                    style={{ width: 320 }}
                    options={models.map(m => ({ value: m.id, label: m.name }))}
                />
            </div>

            <Row gutter={[32, 32]}>
                <Col xs={24} md={12}>
                    <Card className="glass-panel h-full">
                        <Title level={4}><InboxOutlined /> Image Input</Title>
                        <Dragger
                            className="upload-dragger"
                            beforeUpload={() => false}
                            onChange={handleUpload}
                            showUploadList={false}
                        >
                            {preview ? (
                                <img src={preview} alt="preview" style={{ maxHeight: '300px', maxWidth: '100%', borderRadius: '8px' }} />
                            ) : (
                                <div className="upload-placeholder-wrapper">
                                    <p className="ant-upload-text upload-placeholder-text">Click or drag file to this area to upload</p>
                                </div>
                            )}
                        </Dragger>

                        <div className="mt-40">
                            <Title level={4}>Your Question</Title>
                            <TextArea
                                rows={4}
                                style={{ fontSize: '16px' }}
                                value={question}
                                onChange={e => setQuestion(e.target.value)}
                                placeholder="e.g., Is the cat sitting on a striped rug?"
                                className="input-dark"
                            />
                            <Button
                                type="primary"
                                className="neon-button mt-20 w-full"
                                icon={<SendOutlined />}
                                onClick={handlePredict}
                                loading={loading}
                            >
                                Get Answer
                            </Button>
                        </div>
                    </Card>
                </Col>

                <Col xs={24} md={12}>
                    {result && (
                        <div style={{ gap: '20px', display: 'flex', flexDirection: 'column' }}>
                            <Card className="glass-panel">
                                <Title level={4}>Model Answer</Title>
                                <Title level={2} style={{ color: '#00e5ff', margin: '10px 0' }}>{result.answer}.</Title>
                                <div className="flex-between-center">
                                    <Text style={{ fontSize: '18px' }}>Confidence: {Math.round(result.confidence * 100)}%</Text>
                                    <Tag color="cyan">{models.find(m => m.id === selectedModel)?.name || selectedModel}</Tag>
                                </div>
                            </Card>

                            <Card className="glass-panel">
                                <Title level={4}>Candidate Answers</Title>
                                <List
                                    dataSource={result.candidate_answers}
                                    renderItem={item => (
                                        <List.Item>
                                            <div className="w-full">
                                                <div className="flex-between-center"
                                                    style={{ marginBottom: 8, fontSize: 16, fontWeight: 500, letterSpacing: '0.5px' }}>
                                                    <Text style={{ fontSize: '20px' }}>{item.text}</Text>
                                                    <Text style={{ fontSize: '20px', marginLeft: '20px' }}>{Math.round(item.confidence * 100)}%</Text>
                                                </div>
                                                <Progress percent={Math.round(item.confidence * 100)} showInfo={false} strokeColor="#333" trailColor="#111" />
                                            </div>
                                        </List.Item>
                                    )}
                                />
                            </Card>
                        </div>
                    )}
                    {!result && (
                        <div className="empty-result-placeholder">
                            <Text style={{ color: '#444' }}>Model output will appear here</Text>
                        </div>
                    )}
                </Col>
            </Row>
        </div>
    );
};

export default VQAPlayground;
