# Данные

Источник из `projects.md` — [Ames Housing на OpenML](https://www.openml.org/search?exact_name=ames-housing&order=asc&sort=version&status=any&type=data). Перед использованием зафиксируйте OpenML dataset ID/version и проверьте лицензию конкретной версии. Исходные данные не распространяются в этом репозитории.

Код ожидает CSV с `SalePrice` и колонками, перечисленными в `ames_housing.data`. При необходимости названия колонок приводятся к этому контракту отдельным, документированным шагом до запуска. Локальные CSV игнорируются Git.

```bash
make smoke
```

Команда создаёт детерминированный **синтетический smoke-набор**. Он проверяет исполнение и не заменяет Ames Housing.
