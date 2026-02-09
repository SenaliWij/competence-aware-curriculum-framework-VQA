import { useState } from 'react';
import { Row, Col, Input, Button, Card, Typography, Upload, Progress, List, Tag } from 'antd';
import { InboxOutlined, SendOutlined } from '@ant-design/icons';
import { predictVQA } from '../services/api';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Dragger } = Upload;

const VQAPlayground = () => {
    const [question, setQuestion] = useState('');
    const [image, setImage] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [preview, setPreview] = useState('');

    const handlePredict = async () => {
        if (!question) return;
        setLoading(true);
        try {
            const res = await predictVQA(question, image);
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
            <Text className="text-secondary mb-40" style={{ display: 'block' }}>Interact with trained VQA models. Upload an image, ask a question, and understand the model's reasoning process.</Text>

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
                                    <Text>Confidence: {Math.round(result.confidence * 100)}%</Text>
                                    <Tag color="cyan">ViLT Model</Tag>
                                </div>
                            </Card>

                            <Card className="glass-panel">
                                <Title level={4}>Candidate Answers</Title>
                                <List
                                    dataSource={result.candidate_answers}
                                    renderItem={item => (
                                        <List.Item>
                                            <div className="w-full">
                                                <div className="flex-between-center mb-20" style={{ marginBottom: '5px' }}>
                                                    <Text>{item.text}</Text>
                                                    <Text>{Math.round(item.confidence * 100)}%</Text>
                                                </div>
                                                <Progress percent={Math.round(item.confidence * 100)} showInfo={false} strokeColor="#333" trailColor="#111" />
                                            </div>
                                        </List.Item>
                                    )}
                                />
                            </Card>

                            <Card className="glass-panel">
                                <Title level={4}>Reasoning Trace</Title>
                                <List
                                    dataSource={result.reasoning_trace}
                                    renderItem={item => (
                                        <List.Item>
                                            <Text className="text-secondary">&gt; {item}</Text>
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
