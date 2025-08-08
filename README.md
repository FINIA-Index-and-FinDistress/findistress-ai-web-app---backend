# Financial Distress Prediction API

A sophisticated Machine Learning backend for predicting financial distress in companies using LightGBM models with SHAP-based feature importance analysis.

## 🚀 Features

### Core Functionality
- **ML-Powered Predictions**: Advanced LightGBM models for AFR and ROW regions
- **SHAP Analysis**: Detailed feature importance and interpretability
- **Risk Categorization**: Intelligent risk assessment with actionable recommendations
- **Batch Processing**: Efficient bulk prediction capabilities

### Security & Authentication
- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Role-Based Access Control**: User and admin roles with scoped permissions
- **Rate Limiting**: API rate limiting to prevent abuse
- **Audit Logging**: Comprehensive audit trail for security compliance

### Analytics & Insights
- **Dashboard Analytics**: Comprehensive admin dashboard with market comparisons
- **Prediction History**: Detailed tracking and analysis of predictions
- **Export Capabilities**: CSV and JSON export options
- **Model Performance Tracking**: Real-time model performance monitoring

### Data Management
- **PostgreSQL Database**: Robust data storage with connection pooling
- **Data Validation**: Comprehensive input validation and sanitization
- **Quality Assurance**: Prediction quality scoring and validation

## 🏗️ Architecture

```
app/
├── api/
│   └── endpoints.py          # FastAPI route handlers
├── auth/
│   └── security.py           # Authentication & security utilities
├── database/
│   ├── database.py           # Database connection & session management
│   └── models.py             # SQLModel database models
├── models/
│   ├── ml_models.py          # ML model management
│   └── schemas.py            # Pydantic schemas
├── services/
│   └── prediction_service.py # Business logic for predictions
└── main.py                   # FastAPI application entry point
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- PostgreSQL 12+
- Redis (optional, for advanced caching)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd financial-distress-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment configuration**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Database setup**
```bash
# Create PostgreSQL database
createdb financial_distress_db

# The application will automatically create tables on startup
```

6. **Prepare ML models**
```bash
# Place your trained models in app/ml_pipeline/trained_models/
# Required files:
# - afr_best_pipeline.joblib
# - row_best_pipeline.joblib
```

7. **Start the application**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePassword123",
  "full_name": "John Doe"
}
```

#### Login
```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

username=johndoe&password=SecurePassword123
```

### Prediction Endpoints

#### Create Prediction
```http
POST /api/v1/predictions
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "stra_sector": "Manufacturing",
  "wk14": 8.0,
  "car1": 12.0,
  "region": "ROW",
  "Fin_bank": 45.0,
  "Fin_equity": 30.0,
  "Exports": 25.0,
  "GDP": 3.2,
  "PRIME": 5.5
}
```

#### Get Prediction History
```http
GET /api/v1/predictions/history?skip=0&limit=50&risk_category=High
Authorization: Bearer <access_token>
```

### Analytics Endpoints

#### Dashboard Data (Admin Only)
```http
GET /api/v1/analytics/dashboard
Authorization: Bearer <admin_access_token>
```

#### Export Predictions
```http
GET /api/v1/analytics/export?format=csv&start_date=2024-01-01
Authorization: Bearer <access_token>
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `AUTH_SECRET_KEY` | JWT signing key | Required |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token expiry | 30 |
| `REFRESH_TOKEN_EXPIRE_MINUTES` | Refresh token expiry | 10080 |
| `MODEL_BASE_PATH` | ML models directory | `app/ml_pipeline/trained_models` |
| `TRAINING_DATA_PATH` | Training data file path | `data/DF_2025.xlsx` |

### Database Schema

The application automatically creates the following tables:
- `users` - User accounts and authentication
- `prediction_logs` - Prediction history and results
- `influencing_factors` - SHAP feature importance data
- `audit_logs` - Security and compliance audit trail
- `system_metrics` - Performance monitoring data

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"stra_sector": "Manufacturing", "wk14": 5, "car1": 3}'
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use environment-specific configuration
- Set up proper logging and monitoring
- Configure SSL/TLS certificates
- Implement backup strategies for database
- Set up CI/CD pipelines
- Configure load balancing for high availability

## 📈 Monitoring & Maintenance

### Health Checks
- `/health` - Basic health status
- `/health/detailed` - Comprehensive system status (admin only)

### Performance Monitoring
- Database connection pool monitoring
- ML model performance tracking
- API response time metrics
- Rate limiting statistics

### Security Considerations
- Regular security updates
- Audit log review
- Access pattern monitoring
- Token rotation policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📝 API Response Examples

### Successful Prediction Response
```json
{
  "financial_distress_probability": 0.234,
  "model_confidence": 0.892,
  "risk_category": "Medium",
  "financial_health_status": "Financial Health: Moderate",
  "risk_level_detail": "Medium Risk - Monitor Key Indicators",
  "analysis_message": "Company shows moderate financial health with some potential risk factors. Regular monitoring and proactive risk management are recommended.",
  "key_influencing_factors": [
    {
      "name": "Bank Financing Percentage",
      "impact_level": "High",
      "weight": 0.156,
      "description": "Bank financing increases distress risk. High reliance on bank financing may indicate limited access to other funding sources."
    },
    {
      "name": "Export Share",
      "impact_level": "Medium",
      "weight": 0.089,
      "description": "Export share decreases distress risk. International market exposure provides revenue diversification."
    }
  ],
  "recommendations": [
    "Strengthen financial monitoring and reporting systems",
    "Diversify revenue streams and customer base",
    "Review bank financing strategy and optimize financing mix"
  ],
  "created_at": "2024-08-04T10:30:00Z"
}
```

### Dashboard Analytics Response
```json
{
  "user_distress_distribution": [
    {"status": "Distressed", "count": 15, "percentage": 23.4},
    {"status": "Non-Distressed", "count": 49, "percentage": 76.6}
  ],
  "market_distress_distribution": [
    {"status": "Distressed", "count": 1250, "percentage": 15.2},
    {"status": "Non-Distressed", "count": 6980, "percentage": 84.8}
  ],
  "top_risk_factors": [
    {"factor": "Financial Leverage", "impact": "High", "average_weight": 0.145},
    {"factor": "Liquidity Ratio", "impact": "High", "average_weight": 0.132}
  ],
  "ml_insights": {
    "overall_distress_rate": 15.2,
    "user_distress_rate": 23.4,
    "predictions_made": 64,
    "model_accuracy": 92.5,
    "companies_analyzed": 8230
  }
}
```

## 🔍 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Verify connection string
psql "postgresql://username:password@localhost:5432/database_name"

# Check database exists
psql -l | grep financial_distress
```

#### Model Loading Errors
```bash
# Verify model files exist
ls -la app/ml_pipeline/trained_models/

# Check file permissions
chmod 644 app/ml_pipeline/trained_models/*.joblib

# Validate model files
python -c "import joblib; print(joblib.load('path/to/model.joblib'))"
```

#### Authentication Issues
```bash
# Verify JWT secret key is set
echo $AUTH_SECRET_KEY

# Check token expiry settings
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=test&password=test"
```

### Logging

Enable detailed logging for debugging:
```python
# In main.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

View application logs:
```bash
tail -f app.log
```

## 🔐 Security Best Practices

### Production Security Checklist
- [ ] Use strong, unique `AUTH_SECRET_KEY`
- [ ] Enable HTTPS with valid SSL certificates
- [ ] Implement proper CORS configuration
- [ ] Set up rate limiting and DDoS protection
- [ ] Regular security updates and patches
- [ ] Database encryption at rest
- [ ] Secure backup and recovery procedures
- [ ] Monitor and audit access logs
- [ ] Implement proper input validation
- [ ] Use environment variables for secrets

### Database Security
```sql
-- Create dedicated database user
CREATE USER api_user WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE financial_distress_db TO api_user;
GRANT USAGE ON SCHEMA public TO api_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO api_user;
```

## 📊 Performance Optimization

### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_prediction_user_created ON prediction_logs(user_id, created_at);
CREATE INDEX idx_prediction_risk_category ON prediction_logs(risk_category);
CREATE INDEX idx_user_active ON users(is_active) WHERE is_active = true;

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM prediction_logs WHERE user_id = 1;
```

### Application Performance
- Use connection pooling for database connections
- Implement caching for frequently accessed data
- Optimize ML model loading and inference
- Monitor memory usage and garbage collection
- Use async/await for I/O operations

### Monitoring Setup
```python
# Example monitoring with Prometheus
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start_time)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    return response
```

## 🧩 Extensions and Integrations

### Adding New ML Models
```python
# In ml_models.py
def load_custom_model(model_path: str):
    """Load custom model with validation."""
    model = joblib.load(model_path)
    # Add validation logic
    return model

# Register new model
model_manager.register_model("CUSTOM", "/path/to/custom_model.joblib")
```

### Third-Party Integrations
- **Slack Notifications**: Alert on high-risk predictions
- **Email Alerts**: Automated reporting for administrators
- **Data Warehouses**: Export predictions to BigQuery, Snowflake
- **Monitoring Tools**: Integration with DataDog, New Relic
- **CI/CD**: GitHub Actions, Jenkins for automated deployment

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

### Research Papers
- "Financial Distress Prediction using Machine Learning" - [Link]
- "SHAP: A Unified Approach to Explaining ML Predictions" - [Link]
- "LightGBM: A Highly Efficient Gradient Boosting Framework" - [Link]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For technical support or questions:
- Create an issue on GitHub
- Contact the development team
- Review the troubleshooting guide above

---

**Built with ❤️ for the international research community**

*This API is designed for academic and research purposes. Please ensure compliance with your organization's data governance policies when using in production environments.*