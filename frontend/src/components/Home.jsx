import { Button, Typography, Row, Col, Card } from 'antd';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import flatTrainingImg from '../assets/flat_training.jpg';
import curriculumLearningImg from '../assets/curriculum_training.jpg';

const { Title, Paragraph } = Typography;

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="home-container">

            {/* Hero Section */}
            <Row justify="center">
                <Col xs={24} md={18}>
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                        className="text-center"
                    >
                        <Title level={1} className="home-hero-title">
                            DyCDC: Curriculum Learning <br />
                            <span className="gradient-text">for Visual Reasoning</span>
                        </Title>

                        <Paragraph className="home-hero-text">
                            Enhance your VQA models with a curriculum-based training strategy to improve compositional reasoning.
                            By moving beyond flat training, DyCDC empowers transformer-based VQA models to master compositional logic through structured learning.
                        </Paragraph>

                        <div className="flex-center-gap">
                            <Button
                                type="primary"
                                size="large"
                                onClick={() => navigate('/models')}
                                className="neon-button"
                                style={{ height: '50px', padding: '0 40px', fontSize: '1.1rem' }}
                            >
                                See how it works
                            </Button>
                        </div>
                    </motion.div>
                </Col>
            </Row>

            {/* Flat vs Curriculum Section */}
            <Row className="home-section-margin" justify="center">
                <Col xs={24}>
                    <Title level={2} className="text-center" style={{ marginBottom: 8 }}>
                        Flat Training vs Curriculum Learning
                    </Title>
                    <Paragraph className="home-section-text">
                        Why curriculum learning leads to smarter, more reliable visual reasoning models.
                    </Paragraph>
                </Col>
            </Row>

            <Row gutter={[48, 48]}>
                <Col xs={24} md={12}>
                    <motion.div
                        initial={{ opacity: 0, x: -40 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6 }}
                        viewport={{ once: true }}
                    >
                        <Card className="glass-panel" bordered={false}>
                            <div className="home-image-container">
                                <img src={flatTrainingImg} alt="Flat Training" className="img-cover" />
                            </div>
                            <Title level={3}>Flat Training</Title>
                            <Paragraph className="text-secondary" style={{ fontSize: '1.15rem', lineHeight: '1.8' }}>
                                A training approach where the model is exposed to all samples and difficulty levels simultaneously, in no particular order.
                            </Paragraph>
                        </Card>
                    </motion.div>
                </Col>

                <Col xs={24} md={12}>
                    <motion.div
                        initial={{ opacity: 0, x: 40 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6 }}
                        viewport={{ once: true }}
                    >
                        <Card className="glass-panel" bordered={false}>
                            <div className="home-image-container">
                                <img src={curriculumLearningImg} alt="Curriculum Learning" className="img-cover" />
                            </div>
                            <Title level={3}>Curriculum Learning</Title>
                            <Paragraph className="text-secondary" style={{ fontSize: '1.15rem', lineHeight: '1.8' }}>
                                A training approach where training samples are ordered by difficulty, starting simple and progressively introducing harder concepts as the model improves.
                            </Paragraph>
                        </Card>
                    </motion.div>
                </Col>
            </Row>

            {/* How to Start Section */}
            <Row justify="center" className="mt-140 mb-40">
                <Col xs={24}>
                    <Title level={2} className="text-center">How to start?</Title>
                </Col>
            </Row>

            <Row gutter={[32, 32]}>
                <Col xs={24} md={8}>
                    <Card className="glass-panel" bordered={false} hoverable>
                        <Title level={3}>1. Select & Download</Title>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.1rem', lineHeight: '1.8' }}>
                            Browse curriculum-trained model variants. Download whichever model that fits your pipeline.
                        </Paragraph>
                    </Card>
                </Col>

                <Col xs={24} md={8}>
                    <Card className="glass-panel" bordered={false} hoverable>
                        <Title level={3}>2. Understand the Training Strategy</Title>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.1rem', lineHeight: '1.8' }}>
                            Dig into the training strategy, compositional metrics, and architecture decisions behind every model in the repository.
                        </Paragraph>
                    </Card>
                </Col>

                <Col xs={24} md={8}>
                    <Card className="glass-panel" bordered={false} hoverable>
                        <Title level={3}>3. Test & Compare</Title>
                        <Paragraph className="text-secondary" style={{ fontSize: '1.1rem', lineHeight: '1.8' }}>
                            Upload an image, ask a question, and see both models answer side by side. See exactly where curriculum training pulls ahead.
                        </Paragraph>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default Home;
