# MBA_USP_DATA_SCIENCE_2026

RESUMO:


 A adoção de modelos preditivos no diagnóstico oncológico frequentemente esbarra no desbalanceamento severo de classes, cenário no qual técnicas tradicionais de balanceamento degradam a calibração probabilística. O trabalho objetivou avaliar o impacto da utilização de algoritmos de aprendizado não supervisionado como técnica de engenharia de atributos para melhorar o desempenho de classificadores supervisionados (Regressão Logística e árvore de decisão) no diagnóstico de tumores.
 
 A metodologia extraiu pontuações de anomalia e agrupamento via K-Means, Modelos de Mistura Gaussiana, erro de reconstrução via Análise de Componentes Principais e Fator de Outlier Local, concatenando-os aos atributos originais. Avaliou-se o desempenho da abordagem híbrida em dados sintéticos e em uma base real de mamografia, utilizando métricas robustas à assimetria, como PR-AUC e F1-Score. Os resultados demonstraram que a adição de variáveis não supervisionadas aprimorou substancialmente a Regressão Logística nos dados reais, gerando aumentos de 29,1% no F1-Score e 20,2% no PR-AUC em relação à linha de base, superando o método SMOTE sem comprometer a calibração probabilística. Em contrapartida, o modelo de árvores de decisão não apresentou ganhos significativos, indicando redundância, pois sua estrutura já captura relações não lineares internamente. Concluiu-se que o enriquecimento do espaço de atributos com representações não supervisionadas configurou-se como uma estratégia promissora e adaptativa para a otimização de modelos lineares em contextos de alto desbalanceamento, conciliando eficiência preditiva e interpretabilidade.



Referências:
 
Bishop, C.M. 2006. Pattern Recognition And Machine Learning. Springer, New York, NY, USA.

Bose, I.; Chen, X. 2009. Hybrid models using unsupervised clustering for prediction of customer churn. Journal Of Organizational Computing And Electronic Commerce.

Breunig, M.M.; Kriegel, H.P.; Ng, R.T.; Sander, J. 2000. Lof: identifying density-based local outliers. In: ACM SIGMOD International Conference On Management Of Data, 2000, Dallas, TX, USA. Anais... p. 93-104.

Brownlee, J. 2021. Imbalanced Classification With Python: Choose Better Metrics, Balance Skewed Classes, And Apply Cost-Sensitive Learning. 1.3ed. Machine Learning Mastery.

Chawla, N.V.; Bowyer, K.W.; Hall, L.O.; Kegelmeyer, W.P. 2002. Smote: synthetic minority over-sampling technique. Journal Of Artificial Intelligence Research 16: 321-357.

Fávero, L.P.; Belfiore, P. 2017. Manual De Análise De Dados: Estatística E Modelagem Multivariada Com Excel®, SPSS® E SAS®. LTC, Rio de Janeiro, Brasil.

Herreros-Martínez, A.; Magdalena-Benedicto, R.; Vila-Francés, J.; Serrano-López, A.J.; Pérez-Díaz, S.; Mártinez-Herráiz, J.J. 2025. Applied machine learning to anomaly detection in enterprise purchase processes: a hybrid approach using clustering and isolation forest. Information 16(3): 177.

Krawczyk, B. 2016. Learning from imbalanced data: open challenges and future directions. Progress In Artificial Intelligence.
Lloyd, S.P. 1982. Least squares quantization in PCM. IEEE Transactions On Information Theory 28(2): 129-137.

MacQueen, J.B. 1967. Some methods for classification and analysis of multivariate observations. In: Berkeley Symposium On Mathematical Statistics And Probability, 1967, Berkeley, CA, USA. Anais... p. 281-297.

Patel, A. 2019. Hands-On Unsupervised Learning Using Python. O’Reilly Media.

Rajoub, B. 2020. Characterization of biomedical signals: feature engineering and extraction. p. 29-50. In: Biomedical Signal Processing And Artificial Intelligence In Healthcare. Elsevier, [S. l.].

Saito, T.; Rehmsmeier, M. 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. Plos One 10(3): e0118432.

Vluymans, S. 2019. Dealing With Imbalanced And Weekly Labelled Data In Machine Learning Using Fuzzy And Rough Set Methods. Springer, Warsaw, Poland.

Woods, K.S.; Doss, C.C.; Bowyer, K.W.; Solka, J.L.; Priebe, C.E.; Kegelmeyer Jr., W.P. 1993. Comparative evaluation of pattern recognition techniques for detection of microcalcifications in mammography. International Journal Of Pattern Recognition And Artificial Intelligence 7(6): 1417-1436.

