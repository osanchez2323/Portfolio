# Oscar Sanchez — Data Analytics Portfolio

Welcome to my data analytics portfolio. This repository contains projects spanning
data engineering, sports analytics, business intelligence, and data integration —
reflecting the work I do professionally and the analyses I build out of personal interest.

---

## About Me

**Oscar Sanchez** · Senior Legal Operations Analyst · Los Angeles, CA

I specialize in data visualization, business intelligence, and ETL pipeline engineering.
My day-to-day work centers on Python, Tableau, and Excel — building automated pipelines,
interactive dashboards, and recurring reports that give teams reliable visibility into
what's happening and why.

Outside of work, I apply the same analytical instincts to sports analytics — building
player efficiency models, win probability engines, and shot zone breakdowns for NBA data.

📧 osanchez2323@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/oscarsanchez23/)
🌐 [Portfolio Website](https://osanchez2323.github.io)

---

## Projects

### 🏗️ Data Engineering & ETL

| Project | Description | Tools |
|---------|-------------|-------|
| [NBA Data Pipeline](./nba-data-pipeline/) | Production-grade ETL pipeline: NBA Stats API → BigQuery via extract, validate, transform, and load stages. Includes Airflow DAG, data quality checks, and full test suite. | Python · BigQuery · Airflow · pandas |

---

### 🏀 Sports Analytics

| Project | Description | Tools |
|---------|-------------|-------|
| [NBA Player Efficiency Dashboard](./nba-player-efficiency/) | Multi-metric player comparison across PER, TS%, BPM, VORP, and WS/48 for current NBA stars. Includes radar chart and percentile rankings. | Python · Chart.js · pandas |
| [Real-Time Win Probability Model](./nba-win-probability/) | Logistic regression win probability engine that updates live on score margin, game clock, possession, and pace. Includes leverage index and clutch zone detection. | Python · scikit-learn · JavaScript |

---

### 🍽️ Data Integration

| Project | Description | Tools |
|---------|-------------|-------|
| [Food & Travel Multi-Source Integration](./food-travel-integration/) | Merges restaurant, hotel, and travel review data from three distinct source systems with mismatched schemas. Demonstrates entity resolution, deduplication, and schema reconciliation. | Python · pandas · SQL |

---

## Repository Structure

```
Portfolio/
├── README.md                        ← You are here
├── nba-data-pipeline/               ← ETL pipeline project
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   ├── config/
│   ├── src/
│   │   ├── extract/
│   │   ├── validate/
│   │   ├── transform/
│   │   ├── load/
│   │   └── utils/
│   ├── sql/
│   ├── airflow/
│   ├── tests/
│   └── docs/
├── nba-player-efficiency/           ← Coming soon
├── nba-win-probability/             ← Coming soon
└── food-travel-integration/         ← Coming soon
```

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Languages** | Python · SQL · R |
| **Data Engineering** | ETL pipeline design · BigQuery · GCS · Apache Airflow · pandas · dbt |
| **Visualization & BI** | Tableau · Chart.js · D3.js · matplotlib · Excel |
| **Modeling & Statistics** | Logistic regression · Statistical validation · Derived metric design |
| **Data Quality** | Schema enforcement · Null/range checks · Quarantine patterns · Great Expectations |
| **Testing** | pytest · unit testing · mock HTTP (responses) · coverage |
| **Workflow** | Git · GitHub · Jupyter · VS Code |

---

## Getting Started

Each project folder contains its own `README.md` with setup instructions,
dependencies, and usage details. To get started with a specific project:

```bash
# Clone the repository
git clone https://github.com/osanchez2323/Portfolio.git
cd Portfolio

# Navigate to a specific project
cd nba-data-pipeline

# Follow the project-specific README
```

---

## License

All projects in this repository are licensed under the MIT License.
See individual project folders for details.
