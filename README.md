# HW1

Датасет — Facebook Comment Volume Dataset

Датасет слишком большой, гитхаб не разрешает его класть сюда.
Для запуска кода нужно скачать архив [отсюда](https://drive.google.com/file/d/1BZ08fAAY4_6Pi5QTpq668gFj0nsIWZji/view?usp=sharing) и развернуть его в `./hw1`

Предсказать, сколько комментариев наберёт пост. Задача предполагает реализацию градиентного спуска и подсчёта метрик оценки качества модели. Можно использовать линейную алгебру, всякую другую математику готовую в либах.  

Этапы решения:  
— нормировка значений фичей;  
— кросс-валидация по пяти фолдам и обучение линейной регрессии;  
— подсчёт R^2 (коэффициента детерминации) и RMSE.  
Результаты можно оформить в виде следующей таблицы. T1,..T5 — фолды, E — среднее, STD — дисперсия, R^2/RMSE-test — значение соответствующей метрики на тестовой выборке, -train — на обучающей выборке, f0,..fn — значимость признаков (они же переменные, они же фичи).

## Результаты
Результат обучения лежит в директории `./hw1/answers`

Метрики:
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

# HW2

Задание: на графе пользователей найти кластера методом Affinity Propagation. Сравнить эффективность полученных кластеров в задаче рекомендации мест (рядом с графом есть лог чекинов). Чекины для части пользователей прячем, а потом строим "рекомендации" на базе топа чекинов из кластера, куда попал пользователь. Качество рекомендации меряем по точности для первых 10 рекомендованных элементов. Если 10 набрать не удалось (в кластер не попало достаточно пользователей с чекинами) каждый недобранный айтем считаем за промах.

Этапы решения:
- реализация алгоритма affinity propogation
- подсчет самых популярных локаций у НЕ спрятанных пользователей одного из кластеров
- подсчет самых популярных локаций у спрятанных пользователей одного из кластеров

Датасет большой, качать [отсюда](https://drive.google.com/file/d/1tBCsVCULAX-WNI5TpbU-uBGTCbOFKuIK/view?usp=sharing) класть сюда `./hw2`

## Результаты

для 1 тыс пользователей получилось разбиение на два кластера

для user 0 (кластер 0)
```
['420315', '19542', '21714', '9410', '9241', '9246', '42732', '33793', '34055', '9191'] - рекомендованные места
['420315', '21714', '18417', '480992', '25151', '9263', '9410', '15326', '23256', '1221889'] - популярные места
0.3 - совпадение
```

для user 1 (кластер 1)
```
['420315', '21714', '164426', '17208', '763638', '53626', '26919', '412917', '9371', '330817'] - рекомендованные места
['1500177', '1493267', '1441698', '1436795', '1431949', '1423291', '1422219', '1414779', '1404455', '1399686'] - популярные места
0.0 - совпадение
```

для user 2 (кластер 0)
```
['420315', '19542', '21714', '9410', '9241', '9246', '42732', '33793', '34055', '9191'] - рекомендованные места
['59838', '31248', '10184', '1014659', '197128', '83566', '239548', '17888', '10747', '1193153'] - популярные места
0.0 - совпадение
```

вычисления довольно трудоемкие, поэтому сначала было посчитано разбиение на кластеры, а затем рекомендации. Разбиение на кластеры хранится в `./hw2/answers/ap.json`

# HW3
```
Train Fold 0
Test fold 1
RMSE 1.1491350584991595
Test fold 2
RMSE 1.1545439813819531
Test fold 3
RMSE 1.15364404131489
Train Fold 1
Test fold 0
RMSE 1.1478669395588155
Test fold 2
RMSE 1.1540784975566614
Test fold 3
RMSE 1.153178552736331
Train Fold 2
Test fold 0
RMSE 1.1403261226501233
Test fold 1
RMSE 1.1411301972313366
Test fold 3
RMSE 1.1416962321001476
Train Fold 3
Test fold 0
RMSE 1.1405967891509188
Test fold 1
RMSE 1.1414008762965455
Test fold 2
RMSE 1.1428698433010738
```
