| Variant | MI 24h | MI 48h | CA 24h | CA 48h |
|---|---|---|---|---|
| *Component (drops one expert)* | | | | |
| Persistence expert alone | 426.230 | 348.092 | 2099.402 | 2451.595 |
| Cluster expert alone | 439.654 | 335.829 | 1846.408 | 1757.977 |
| *Backbone (replaces AR(2))* | | | | |
| ARIMA(1,0,1) | 401.602 | 306.698 | 1855.843 | 1825.904 |
| AR(1) | 399.109 | 308.929 | 1849.936 | 1812.507 |
| *Aggregator (replaces the geometric mean)* | | | | |
| Arithmetic mean | 379.670 | 300.834 | 1887.964 | 1957.985 |
| Convex (weight tuned on validation) | 439.654 | 335.829 | 1839.399 | 1763.198 |
| *Weather (replaces the forecast)* | | | | |
| Last observed weather | 404.351 | 309.888 | 1857.704 | 1833.107 |
| **PRiCE (full)** | 402.323 | 309.690 | 1855.494 | 1809.473 |
