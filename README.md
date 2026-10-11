# 🍷 Wine Quality — XGBoost, FastAPI, Streamlit, Docker, AWS

Prédit si un vin est **bon** (qualité ≥ 7) ou **mauvais** à partir de ses caractéristiques
(pH, alcool, type...). Pipeline : doublons → écrêtage 3×IQR (bornes calculées sur le train) →
XGBoost optimisé (RandomizedSearchCV, score F1).

```
Streamlit (UI) ──HTTP──▶ FastAPI (/predict) ──▶ XGBoost (models/wine_model.joblib)
        └── déployés en conteneurs : GitHub Actions ▶ Amazon ECR ▶ ECS Express Mode
```

## Structure

```
app/main.py            API FastAPI (/health, /predict, /model-info)
streamlit_app/         Interface Streamlit + son Dockerfile
tests/                 Tests unitaires (pytest)
train.py               Entraînement → models/wine_model.joblib
Dockerfile             Image de l'API
docker-compose.yml     API + UI en local
.github/workflows/     ci.yml (tests) et cd.yml (build, push ECR, déploiement)
```

---

## Étape 1 — Environnement local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
```

> Entraîne le modèle **dans cet environnement** (pas dans Colab) : le `.joblib` doit être
> chargé avec les mêmes versions de scikit-learn / xgboost que celles du conteneur.

## Étape 2 — Données et entraînement

Place les deux fichiers dans `data/` :

- `data/winequality-red.csv`
- `data/winequality-white.csv`

(Dataset « Wine Quality » de l'UCI, séparateur `;`.)

```bash
python train.py
```

Cela affiche les doublons supprimés, les meilleurs hyperparamètres, les métriques de test
(F1, AUC...) et crée `models/wine_model.joblib`. Compare avec ton notebook
(XGBoost : F1 test ≈ 0.60, AUC ≈ 0.87).

## Étape 3 — Tests et lint en local

```bash
ruff check .
pytest -v
```

Les tests utilisent un petit modèle factice : ils ne nécessitent ni les données ni le vrai modèle.

## Étape 4 — Lancer l'application en local

Sans Docker :

```bash
uvicorn app.main:app --reload                              # API : http://localhost:8000/docs
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py                          # UI : http://localhost:8501
```

Avec Docker :

```bash
docker compose up --build
```

Test rapide de l'API :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "fixed_acidity":7,"volatile_acidity":0.3,"citric_acid":0.3,"residual_sugar":2,
  "chlorides":0.05,"free_sulfur_dioxide":30,"total_sulfur_dioxide":100,"density":0.995,
  "ph":3.2,"sulphates":0.5,"alcohol":12,"wine_type":"white"}'
```

## Étape 5 — Repo GitHub

Crée un dépôt vide sur GitHub, puis :

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<ton-user>/wine-quality.git
git push -u origin main
```

`models/wine_model.joblib` **doit être commité** (il est copié dans l'image Docker).
Les CSV de `data/` sont ignorés par Git. Si le modèle dépasse 100 Mo, utilise Git LFS.

> Le premier push sur `main` déclenche le workflow CD, qui échouera tant que les étapes AWS
> ci-dessous ne sont pas faites. C'est normal : refais un push (ou « Re-run jobs ») ensuite.

## Étape 6 — Préparer AWS (une seule fois)

> ⚠️ **AWS App Runner est fermé aux nouveaux clients depuis le 30 avril 2026.**
> Ce projet utilise donc **Amazon ECS Express Mode** (recommandé par AWS) : on fournit une
> image, il crée le cluster Fargate, le load balancer et une URL HTTPS.

Prérequis : AWS CLI configurée (`aws configure`) avec un utilisateur administrateur.
Remplace `<ACCOUNT_ID>` par ton identifiant de compte à 12 chiffres.

### 6.1 Dépôts ECR

```bash
aws ecr create-repository --repository-name wine-api --region eu-west-3
aws ecr create-repository --repository-name wine-streamlit --region eu-west-3
```

> Si ECS Express Mode n'est pas disponible dans ta région, change `AWS_REGION` dans
> `cd.yml` (par exemple `eu-west-1`) et recrée les dépôts dans cette région.

### 6.2 Rôles IAM pour ECS

Rôle d'exécution (pull ECR + logs) :

```bash
cat > ecs-tasks-trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file://ecs-tasks-trust.json
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

Rôle d'infrastructure (crée load balancer, groupes de sécurité...) :

```bash
cat > ecs-infra-trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name ecsInfrastructureRole \
  --assume-role-policy-document file://ecs-infra-trust.json
aws iam attach-role-policy --role-name ecsInfrastructureRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
```

(Si un de ces rôles existe déjà, ou si une commande échoue, tu peux aussi les faire créer par
la console : ECS → Express mode → « Create », option « Create new role ».)

### 6.3 Autoriser GitHub Actions (OIDC, sans clé d'accès)

1. Console IAM → *Identity providers* → *Add provider* → **OpenID Connect** :
   URL `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
2. Crée le rôle `github-actions-wine` avec cette *trust policy* :

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:<ton-user>/wine-quality:ref:refs/heads/main"}
    }
  }]
}
```

3. Attache-lui `AmazonEC2ContainerRegistryPowerUser`, plus cette politique en ligne :

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ecs:CreateCluster", "ecs:RegisterTaskDefinition",
      "ecs:CreateExpressGatewayService", "ecs:UpdateExpressGatewayService",
      "ecs:DescribeExpressGatewayService", "ecs:DescribeClusters",
      "ecs:DescribeServices", "ecs:ListServiceDeployments",
      "ecs:DescribeServiceDeployments", "ecs:TagResource", "ecs:UntagResource",
      "iam:PassRole"
    ],
    "Resource": "*"
  }]
}
```

### 6.4 Secrets GitHub

*Settings → Secrets and variables → Actions → New repository secret* :

| Secret | Valeur |
|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::<ACCOUNT_ID>:role/github-actions-wine` |
| `ECS_EXECUTION_ROLE_ARN` | `arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole` |
| `ECS_INFRASTRUCTURE_ROLE_ARN` | `arn:aws:iam::<ACCOUNT_ID>:role/ecsInfrastructureRole` |

## Étape 7 — Déployer

Un push sur `main` lance automatiquement :

1. **test** : `ruff` + `pytest` ;
2. build des deux images Docker, tag = SHA du commit, push sur ECR ;
3. déploiement de **wine-api** (port 8000, health check `/health`) ;
4. déploiement de **wine-streamlit** (port 8501), avec `API_URL` pointant vers l'API.

L'onglet *Actions* de GitHub affiche, dans le résumé du job, l'URL de l'API (`/docs`) et celle
de Streamlit. Le premier déploiement prend quelques minutes (création du load balancer).

## Étape 8 — Tester l'application déployée

```bash
curl https://<url-api>/health
curl -X POST https://<url-api>/predict -H "Content-Type: application/json" -d '{ ... }'
```

Ouvre l'URL Streamlit, saisis alcool, pH, type, etc., puis clique sur **Prédire**.

## Étape 9 — Nettoyage (après la soutenance)

ECS Express Mode crée des ressources facturées (tâches Fargate, load balancer). Supprime-les :

```bash
aws ecs delete-express-gateway-service --service-arn <arn-wine-api> --region eu-west-3
aws ecs delete-express-gateway-service --service-arn <arn-wine-streamlit> --region eu-west-3
aws ecr delete-repository --repository-name wine-api --force --region eu-west-3
aws ecr delete-repository --repository-name wine-streamlit --force --region eu-west-3
```

(Les ARN sont visibles dans la console ECS.) Vérifie ensuite dans la console que le load
balancer a bien disparu.

---

## Limites à mentionner dans le rapport

- Classes déséquilibrées (~19 % de bons vins) : F1 test ≈ 0.60, donc résultat **indicatif**.
- Écrêtage et split faits sur le dataset dédoublonné ; bornes calculées sur le train seul.
- Seuil de décision à 0.5 (comme dans le notebook) ; ajustable dans `app/main.py`.
- Les écarts entre XGBoost, Random Forest et SVM sont du même ordre que la variabilité de la CV.
