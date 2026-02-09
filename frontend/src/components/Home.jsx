import { Button, Typography, Row, Col, Card } from 'antd';
import { RocketOutlined, ThunderboltOutlined, ExperimentOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import flatTrainingImg from "../assets/flat_training.jpg";
import curriculumLearningImg from "../assets/curriculum_training.jpg";



const { Title, Paragraph } = Typography;

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="home-container">

            {/* ================= HERO SECTION (NO IMAGE) ================= */}
            <Row justify="center">
                <Col xs={24} md={18}>
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                        className="text-center"
                    >
                        <Title level={1} className="home-hero-title">
                            VQA-CL: Redefining <br />
                            <span className="gradient-text">Visual Reasoning</span>
                        </Title>

                        <Paragraph className="home-hero-text">
                            Empower your VQA models to understand the world with compositional reasoning,
                            overcoming complex visual challenges through structured curriculum learning.
                        </Paragraph>

                        <div className="flex-center-gap">
                            <Button
                                type="primary"
                                size="large"
                                onClick={() => navigate('/config')}
                                className="neon-button"
                                style={{ height: '50px', padding: '0 40px', fontSize: '1.1rem' }}
                            >
                                Explore Framework
                            </Button>
                        </div>
                    </motion.div>
                </Col>
            </Row>

            {/* ================= FLAT vs CURRICULUM SECTION ================= */}
            <Row className="home-section-margin" justify="center">
                <Col xs={24}>
                    <Title
                        level={2}
                        className="text-center mb-20"
                    >
                        Flat Training vs Curriculum Learning
                    </Title>

                    <Paragraph className="home-section-text">
                        Why structured learning leads to smarter, more reliable visual reasoning models.
                    </Paragraph>
                </Col>
            </Row>

            <Row gutter={[48, 48]}>
                {/* Flat Training */}
                <Col xs={24} md={12}>
                    <motion.div
                        initial={{ opacity: 0, x: -40 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6 }}
                        viewport={{ once: true }}
                    >
                        <Card className="glass-panel" bordered={false}>
                            {/* Image Placeholder */}
                            <div className="home-image-container">
                                <img
                                    src={flatTrainingImg}
                                    alt="Flat Training"
                                    className="img-cover"
                                />
                            </div>


                            <Title level={3}>Flat Training</Title>
                            <Paragraph className="text-secondary">
                                Models are trained on all question types simultaneously, without any
                                structured learning order. This often leads to shallow pattern matching
                                and poor compositional reasoning.
                            </Paragraph>
                        </Card>
                    </motion.div>
                </Col>

                {/* Curriculum Learning */}
                <Col xs={24} md={12}>
                    <motion.div
                        initial={{ opacity: 0, x: 40 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6 }}
                        viewport={{ once: true }}
                    >
                        <Card className="glass-panel" bordered={false}>
                            {/* Image Placeholder */}
                            <div className="home-image-container">
                                <img
                                    src={curriculumLearningImg}
                                    alt="Curriculum Learning"
                                    className="img-cover"
                                />
                            </div>

                            <Title level={3}>Curriculum Learning</Title>
                            <Paragraph className="text-secondary">
                                Training progresses from simple concepts to complex reasoning steps.
                                This structured approach enables robust compositional understanding
                                and significantly improves VQA accuracy.
                            </Paragraph>
                        </Card>
                    </motion.div>
                </Col>
            </Row>

            {/* ================= FEATURE CARDS (KEEPING YOUR ORIGINALS) ================= */}
            <Row gutter={[32, 32]} className="mt-140">
                <Col xs={24} md={8}>
                    <Card className="glass-panel" bordered={false} hoverable>
                        <ThunderboltOutlined className="feature-icon" style={{ color: '#00e5ff' }} />
                        <Title level={3}>Seamless Selection</Title>
                        <Paragraph className="text-secondary">
                            Effortlessly browse and select from a curated list of leading VQA models like ViLT and LXMERT.
                        </Paragraph>
                    </Card>
                </Col>

                <Col xs={24} md={8}>
                    <Card className="glass-panel" bordered={false} hoverable>
                        <RocketOutlined className="feature-icon" style={{ color: '#2979ff' }} />
                        <Title level={3}>Curriculum Learning</Title>
                        <Paragraph className="text-secondary">
                            Guide models from simpler tasks to complex reasoning, dramatically enhancing accuracy.
                        </Paragraph>
                    </Card>
                </Col>

                <Col xs={24} md={8}>
                    <Card className="glass-panel" bordered={false} hoverable>
                        <ExperimentOutlined className="feature-icon" style={{ color: '#d500f9' }} />
                        <Title level={3}>Interactive Testing</Title>
                        <Paragraph className="text-secondary">
                            Test your trained models in real-time with our advanced visual playground.
                        </Paragraph>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default Home;
