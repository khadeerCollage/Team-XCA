from backend.ml.ml_routes import _get_driver, _get_session
from backend.ml.train import train_and_evaluate

try:
    print("Connecting to DBs...")
    driver = _get_driver()
    session = _get_session()
    print("DBs connected.")
    print("Starting Training...")
    results = train_and_evaluate(driver, session)
    print("Training Results:", results)
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if 'session' in locals():
        session.close()
    if 'driver' in locals():
        driver.close()
