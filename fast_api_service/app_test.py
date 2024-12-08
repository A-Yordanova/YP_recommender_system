import logging
import requests
from recsys_app.utility_scripts.database import create_postgresql_connection


RECOMMENDATIONS_URL = "http://127.0.0.1:8081"
HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

TEST_USER_NO_PERSONAL_RECS = 123
TEST_USER_WITH_PERSONAL_RECS = 138131
TEST_ITEM_IDS = [216266, 312728, 65273]

logging.basicConfig(filename="test_service.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("-" * 80)
logging.info(f"НАЧАЛО ТЕСТИРОВАНИЯ СЕРВИСА:")
logging.info("-" * 80)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Тестирование сервиса для пользователя без персональных рекомендаций
logging.info(f'Тестируем получение рекомендаций по умолчанию для user_id="{TEST_USER_NO_PERSONAL_RECS}" без персональных рекомендаций.')

params = {"user_id": TEST_USER_NO_PERSONAL_RECS}
resp = requests.post(RECOMMENDATIONS_URL + "/recommendations_offline", headers=HEADERS, params=params)

if resp.status_code == 200:
    default_recs = resp.json()["recs"]
    logging.info(f"Рекомендации по умолчанию получены успешно. Результат:")
    logging.info(default_recs)
else:
    default_recs = []
    logging.error(f"Получение рекомендаций завершилось с ошибкой. Код: {resp.status_code}.")
logging.info("-" * 80)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Тестирование сервиса для пользователя с оффлайн-историей, но без онлайн-истории
logging.info(f'Тестируем получение оффлайн-рекомендаций для user_id="{TEST_USER_WITH_PERSONAL_RECS}" с персональными рекомендациями, но без онлайн-истории.')

# Получение списка последних взаимодействий пользователя
params = {"user_id": TEST_USER_WITH_PERSONAL_RECS}
resp = requests.post(RECOMMENDATIONS_URL + "/get_user_history", headers=HEADERS, params=params)

if resp.status_code == 200:
    logging.info(f"История пользователя получена успешно. Ожидаем пустой список.")
    logging.info(f'История пользователя: {resp.json()["events"]}.')
else:
    result = None
    logging.error(f"Получение истории завершилось с ошибкой. Код: {resp.status_code}.")

# Получение персональных рекомендаций
logging.info(f'Получаем персональные рекомендации для user_id="{TEST_USER_WITH_PERSONAL_RECS}".')

params = {"user_id": TEST_USER_WITH_PERSONAL_RECS}
resp = requests.post(RECOMMENDATIONS_URL + "/recommendations_offline", headers=HEADERS, params=params)

if resp.status_code == 200:
    logging.info(f"Персональные рекомендации получены успешно. Результат:")
    logging.info(resp.json()["recs"])
else:
    recs = []
    logging.error(f"Получение персональных рекомендаций завершилось с ошибкой. Код: {resp.status_code}.")
logging.info("-" * 80)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Тестирование микса рекомендаций для пользователя с онлайн и оффлайн историей
logging.info(f'Тестируем получение микса рекомендаций для user_id="{TEST_USER_WITH_PERSONAL_RECS}" с персональными рекомендациями и с онлайн-историей.')
logging.info(f'Добавляем онлайн-историю для user_id="{TEST_USER_WITH_PERSONAL_RECS}".')

# Добавление событий в историю пользователя и сохранение истории взаимодействий (event_store)

for test_item_id in TEST_ITEM_IDS:
    params = {"user_id": TEST_USER_WITH_PERSONAL_RECS, "item_id": test_item_id}
    resp = requests.post(RECOMMENDATIONS_URL + "/save_event", headers=HEADERS, params=params)
    
    if resp.status_code == 200:
        logging.info(f'Взаимодействие с item_id="{test_item_id}" успешно добавлено в онлайн-историю пользователя.')
    else:
        result = None
        logging.error(f"Добавление событий в онлайн-историю пользователя завершилось с ошибкой. Код: {resp.status_code}.")

# Получение онлайн-истории пользователя (три последних взаимодействия)
logging.info(f'Получаем онлайн-историю (три последних взаимодействия) для user_id="{TEST_USER_WITH_PERSONAL_RECS}".')

params = {"user_id": TEST_USER_WITH_PERSONAL_RECS, "k": 3}
resp = requests.post(RECOMMENDATIONS_URL + "/get_user_history", headers=HEADERS, params=params)

if resp.status_code == 200:
    online_history = resp.json()["events"]
    logging.info(f"Онлайн-история пользователя успешно получена. Результат:")
    logging.info(online_history)
else:
    result = None
    logging.error(f"Получение онлайн-истории пользователя завершилось с ошибкой. Код: {resp.status_code}.")

# Получение микса рекомендаций
logging.info(f'Получаем микс рекомендации для user_id="{TEST_USER_WITH_PERSONAL_RECS}".')

params = {"user_id": TEST_USER_WITH_PERSONAL_RECS, "k": 10}
response_offline = requests.post(RECOMMENDATIONS_URL + "/recommendations_offline", headers=HEADERS, params=params)
recs_offline = response_offline.json()["recs"]
logging.info(f'Подготовленные оффлайн рекомендации для user_id="{TEST_USER_WITH_PERSONAL_RECS}": {recs_offline}')

params={"item_ids": online_history, "k": 10}
response_online = requests.post(RECOMMENDATIONS_URL + "/recommendations_online", headers=HEADERS, params=params)
recs_online = [rec["item_id"] for rec in response_online.json()["recs"]]
logging.info(f'Онлайн (похожие товары) рекомендации для user_id="{TEST_USER_WITH_PERSONAL_RECS}": {recs_online}')

params = {"user_id": TEST_USER_WITH_PERSONAL_RECS, "k": 10}
response_blended = requests.post(RECOMMENDATIONS_URL + "/recommendations", headers=HEADERS, params=params)
recs_blended = response_blended.json()["recs"]
logging.info(f"Информация для проверки результата. Ожидаемый порядок ID: первый ID в списке из оффлайн рекомендаций, второй из онлайн, и так далее по очереди.")
logging.info(f'Смешанный список рекомендаций для user_id="{TEST_USER_WITH_PERSONAL_RECS}": {recs_blended}')
logging.info("-" * 80)
logging.info(f"ТЕСТИРОВАНИЯ СЕРВИСА ЗАВЕРШЕНО.")
logging.info("-" * 80)