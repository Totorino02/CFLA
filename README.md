# CFLA

**Titre : Benchmark des algorithmes de CFL (Clustered Federated Learning)**
### Résumé

L'apprentissage fédéré est une technique d'intelligence artificielle permettant d'entraîner des modèles sur des données décentralisées, sans les partager. Chaque client entraîne le modèle sur ses données locales puis envoie uniquement les mises à jour des paramètres à un serveur central qui les agrège. L'apprentissage fédéré par clusters (Clustered Federated Learning) est une approche qui améliore l'apprentissage fédéré classique (où on entraîne un modèle global) en regroupant les clients similaires. Au lieu d'un modèle global unique, plusieurs modèles sont créés pour différents groupes de clients ayant des caractéristiques semblables, ce qui permet une meilleure personnalisation tout en préservant la confidentialité.
Ce projet vise à implémenter quelques algorithmes de CFL et à faire leur benchmark sur des métriques que nous aurons définies.

### Méta-données
- Projet : Recherche
- Mots clés : Machine Learning, Apprentissage fédéré
- Sujet assigné à : HOUNSI Antoine (hounsi.madouvi.etu@univ-lille.fr)
- Prérequis : Python, PyTorch

### Contexte

Ce projet se situe dans le contexte d'un mémoire de master sur l'apprentissage fédéré.

### Problématique
La problématique est d'implémenter et de benchmarker quelques algorithmes de CFL.

### Travail à effectuer
- Implémenter quelques algorithmes de Clustered Federated Learning
- Définir les métriques de benchmark
- Trouver des bases de données pertinentes sur lesquelles évaluer ces algorithmes
- Évaluer les performances de ces algorithmes et faire le benchmark
- Documenter et discuter les résultats

### Bibliographie
- Sattler, F., Müller, K.-R., & Samek, W. (2019). Clustered federated learning: Model-agnostic distributed multi-task optimization under privacy constraints. arXiv. https://arxiv.org/abs/1910.01991
- Briggs, C., Fan, Z., & Andras, P. (2020). Federated learning with hierarchical clustering of local updates to improve training on non-IID data. arXiv. https://arxiv.org/abs/2004.11791
- Ghosh, A., Chung, J., Yin, D., & Ramchandran, K. (2021). An efficient framework for clustered federated learning. arXiv. https://arxiv.org/abs/2006.04088
- Zhang, Y., Chen, H., Lin, Z., Chen, Z., & Zhao, J. (2025). LCFed: An efficient clustered federated learning framework for heterogeneous data. arXiv. https://arxiv.org/abs/2303.01850


