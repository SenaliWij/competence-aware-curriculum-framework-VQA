import { useState } from 'react';
import { Row, Col, Input, Button, Card, Typography, Upload, Progress, List, Tag, Alert } from 'antd';
import { InboxOutlined, SendOutlined, SwapOutlined, WarningOutlined } from '@ant-design/icons';
import { predictVQA, compareVQA } from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Dragger } = Upload;

// Reusable result card
const ModelResultCard = ({ data, accentColor, label }) => {
    if (!data) return null;

    return (
        <Card
            className="glass-panel"
            style={{ height: '100%', borderTop: `3px solid ${accentColor}` }}
        >
            {/* Header */}
            <div style={{ marginBottom: 16 }}>
                <Text style={{
                    fontSize: '1rem', fontWeight: 600,
                    color: 'var(--text-secondary)',
                    textTransform: 'uppercase', letterSpacing: '0.5px',
                }}>
                    {label}
                </Text>
                <Title level={4} style={{ margin: '4px 0 0', fontSize: '1.15rem', color: accentColor }}>
                    ({data.model_name})
                </Title>
            </div>

            {/* Answer */}
            <div style={{ marginBottom: 16 }}>
                <Text style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>Answer:</Text>
                <Title level={2} style={{ color: '#fff', margin: '4px 0 0', fontSize: '2rem', fontWeight: 700 }}>
                    {data.answer}.
                </Title>
            </div>

            {/* Confidence */}
            <Text style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
                Confidence: <span style={{ color: '#fff', fontWeight: 600 }}>
                    {Math.round(data.confidence * 100)}%
                </span>
            </Text>

            {/* Model tag and  progress bar */}
            <div style={{ margin: '14px 0 20px' }}>
                <Tag style={{
                    fontSize: '0.85rem', padding: '3px 12px',
                    background: `${accentColor}22`, border: `1px solid ${accentColor}55`,
                    color: accentColor, borderRadius: 6,
                }}>
                    {data.model_name}
                </Tag>
                <Progress
                    percent={Math.round(data.confidence * 100)}
                    showInfo={false}
                    strokeColor={accentColor}
                    trailColor="rgba(255,255,255,0.07)"
                    style={{ marginTop: 10 }}
                />
            </div>

            {/* Candidate answers */}
            <List
                dataSource={data.candidate_answers}
                renderItem={item => (
                    <List.Item style={{ borderBottom: '1px solid rgba(255,255,255,0.07)', padding: '10px 0' }}>
                        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between' }}>
                            <Text style={{ fontSize: '1.05rem', fontWeight: 500 }}>{item.text}</Text>
                            <Text style={{ fontSize: '1rem', color: accentColor, fontWeight: 600, marginLeft: 16 }}>
                                ({Math.round(item.confidence * 100)}%)
                            </Text>
                        </div>
                    </List.Item>
                )}
            />
        </Card>
    );
};

const VQAPlayground = () => {
    const [question, setQuestion] = useState('');
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState('');
    const [error, setError] = useState('');

    // Get Answer
    const [singleResult, setSingleResult] = useState(null);
    // Compare
    const [compareResult, setCompareResult] = useState(null);

    const [loadingPredict, setLoadingPredict] = useState(false);
    const [loadingCompare, setLoadingCompare] = useState(false);

    //Shared validation for image and question check
    const validate = (action) => {
        setError('');
        if (!image && !question.trim()) {
            setError(`Please upload an image & enter a question before ${action}.`);
            return false;
        }
        if (!image) {
            setError(`Please upload an image before ${action}.`);
            return false;
        }
        if (!question.trim()) {
            setError(`Please enter a question before ${action}.`);
            return false;
        }
        return true;
    };

    /*Get Answer (proposed model only)*/
    const handlePredict = async () => {
        if (!validate('running inference')) return;

        setLoadingPredict(true);
        setCompareResult(null); // clear compare view
        try {
            const res = await predictVQA(question, image);
            setSingleResult(res);
        } catch (err) {
            console.error(err);
            setError('Inference failed. Make sure the backend is running & try again.');
        }
        setLoadingPredict(false);
    };

    //Compare proposed vs baseline
    const handleCompare = async () => {
        if (!validate('comparing')) return;

        setLoadingCompare(true);
        setSingleResult(null); // clear single view
        try {
            const res = await compareVQA(question, image);
            setCompareResult(res);
        } catch (err) {
            console.error(err);
            setError('Comparison failed. Make sure the backend is running & try again.');
        }
        setLoadingCompare(false);
    };

    const handleUpload = ({ file }) => {
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setError('');
        return false;
    };

    const isLoading = loadingPredict || loadingCompare;

    return (
        <div style={{ padding: '40px 4%', minHeight: '100vh' }}>
            <Title level={2} style={{ marginBottom: 8, fontSize: '2.2rem' }}>
                VQA Testing <span className="text-accent">Interface</span>
            </Title>
            <Paragraph className="text-secondary" style={{ fontSize: '1.15rem', marginBottom: 30 }}>
                Upload an image, ask a question and get an answer or compare both models side by side.
            </Paragraph>

            {/* Validation / API error */}
            {error && (
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
                    style={{
                        marginBottom: 24,
                        background: 'rgba(255, 50, 50, 0.08)',
                        border: '1px solid rgba(255,50,50,0.3)',
                        borderRadius: 8,
                    }}
                />
            )}

            <Row gutter={[28, 28]}>
                {/* Input panel */}
                <Col xs={24} lg={10}>
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
                        >
                            {preview ? (
                                <img
                                    src={preview}
                                    alt="preview"
                                    style={{ maxHeight: '220px', maxWidth: '100%', borderRadius: '8px' }}
                                />
                            ) : (
                                <div style={{ padding: '32px 20px' }}>
                                    <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
                                        Click or drag file to this area to upload
                                    </p>
                                </div>
                            )}
                        </Dragger>

                        <div style={{ marginTop: 28 }}>
                            <Title level={4} style={{ fontSize: '1.25rem', marginBottom: 12 }}>
                                Your Question
                            </Title>
                            <TextArea
                                rows={4}
                                style={{ fontSize: '1.05rem', lineHeight: '1.6' }}
                                value={question}
                                onChange={e => { setQuestion(e.target.value); setError(''); }}
                                placeholder="e.g., Is the cat sitting on a striped rug?"
                                className="input-dark"
                            />

                            {/*Action buttons*/}
                            <Button
                                type="primary"
                                className="neon-button"
                                icon={<SendOutlined />}
                                onClick={handlePredict}
                                loading={loadingPredict}
                                disabled={loadingCompare}
                                style={{
                                    marginTop: 16, width: '100%',
                                    height: 48, fontSize: '1.1rem', fontWeight: 600,
                                }}
                            >
                                {loadingPredict ? 'Getting Answer…' : 'Get Answer'}
                            </Button>

                            <Button
                                icon={<SwapOutlined />}
                                onClick={handleCompare}
                                loading={loadingCompare}
                                disabled={loadingPredict}
                                style={{
                                    marginTop: 10, width: '100%',
                                    height: 44, fontSize: '1rem', fontWeight: 500,
                                    background: 'transparent',
                                    border: '1px solid rgba(255,152,0,0.5)',
                                    color: '#ff9800',
                                }}
                            >
                                {loadingCompare ? 'Comparing…' : 'Compare with Baseline'}
                            </Button>
                        </div>
                    </Card>
                </Col>

                {/* Results panel */}
                <Col xs={24} lg={14}>

                    {/* Single-model result */}
                    {singleResult && (
                        <ModelResultCard
                            data={{
                                model_name: 'VILT-CL',
                                answer: singleResult.answer,
                                confidence: singleResult.confidence,
                                candidate_answers: singleResult.candidate_answers,
                            }}
                            accentColor="#00e5ff"
                            label="Proposed Model"
                        />
                    )}

                    {/* side by side comparison */}
                    {compareResult && (
                        <Row gutter={[20, 20]}>
                            <Col xs={24} md={12}>
                                <ModelResultCard
                                    data={compareResult.proposed}
                                    accentColor="#00e5ff"
                                    label="Proposed Model"
                                />
                            </Col>
                            <Col xs={24} md={12}>
                                <ModelResultCard
                                    data={compareResult.baseline}
                                    accentColor="#ff9800"
                                    label="Baseline Model"
                                />
                            </Col>
                        </Row>
                    )}

                    {/* Placeholder when no results yet */}
                    {!singleResult && !compareResult && (
                        <div style={{
                            height: '100%', minHeight: 400,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            border: '1px dashed rgba(255,255,255,0.12)', borderRadius: 12,
                            flexDirection: 'column', gap: 12,
                        }}>
                            <SendOutlined style={{ fontSize: 36, color: '#333' }} />
                            <Text style={{ color: '#555', fontSize: '1.1rem' }}>
                                Model output will appear here
                            </Text>
                            <Text style={{ color: '#444', fontSize: '0.95rem' }}>
                                Use "Get Answer" or "Compare with Baseline"
                            </Text>
                        </div>
                    )}
                </Col>
            </Row>
        </div>
    );
};

export default VQAPlayground;
