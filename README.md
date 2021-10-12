# HW1

Датасет — Facebook Comment Volume Dataset

Предсказать, сколько комментариев наберёт пост. Задача предполагает реализацию градиентного спуска и подсчёта метрик оценки качества модели. Можно использовать линейную алгебру, всякую другую математику готовую в либах.  

Этапы решения:  
— нормировка значений фичей;  
— кросс-валидация по пяти фолдам и обучение линейной регрессии;  
— подсчёт R^2 (коэффициента детерминации) и RMSE.  
Результаты можно оформить в виде следующей таблицы. T1,..T5 — фолды, E — среднее, STD — дисперсия, R^2/RMSE-test — значение соответствующей метрики на тестовой выборке, -train — на обучающей выборке, f0,..fn — значимость признаков (они же переменные, они же фичи).

## Результаты
```sh
FOLD 1 RMSE ON TEST: 0.02579029556864831
FOLD 1 R^2 ON TEST: 0.10087323842180962
FOLD 1 RMSE ON TRAIN: 0.01530528642800414
FOLD 1 R^2 ON TRAIN: 0.12129745061770347
FOLD 2 RMSE ON TEST: 0.01763531043889485
FOLD 2 R^2 ON TEST: 0.07499701935177949
FOLD 2 RMSE ON TRAIN: 0.015622747336389884
FOLD 2 R^2 ON TRAIN: 0.07493511734072711
FOLD 3 RMSE ON TEST: 0.01762330862597821
FOLD 3 R^2 ON TEST: 0.08561315314991103
FOLD 3 RMSE ON TRAIN: 0.015784203082686743
FOLD 3 R^2 ON TRAIN: 0.082691954191074
FOLD 4 RMSE ON TEST: 0.015521087006985259
FOLD 4 R^2 ON TEST: 0.09255289644978826
FOLD 4 RMSE ON TRAIN: 0.01674915858337463
FOLD 4 R^2 ON TRAIN: 0.08984443111737495
FOLD 5 RMSE ON TEST: 0.0199204529722797
FOLD 5 R^2 ON TEST: 0.022814031749481933
FOLD 5 RMSE ON TRAIN: 0.01633832260344229
FOLD 5 R^2 ON TRAIN: 0.007762244158007081
RMSE ON TEST MEAN: 0.019298090922557264
RMSE ON TEST STD: 0.0035318988927683184
R2 ON TEST MEAN: 0.07537006782455406
R2 ON TEST STD: 0.02761469970920109
RMSE ON TRAIN MEAN: 0.015959943606779537
RMSE ON TRAIN STD: 0.0005175312594796124
R2 ON TRAIN MEAN: 0.07530623948497732
R2 ON TRAIN STD: 0.03726507301882258
```
