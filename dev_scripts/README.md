# Database Scripts

All scripts should be executed from the root directory.

Upload datasets (csv-files) to DB:
```bash
python dev_scripts/db_upload_data.py
```

Show first row of the table:
```bash
python dev_scripts/db_show_table.py "table_name"
```

Delete table:
```bash
python dev_scripts/db_delete_table.py "table_name"
```

Launch Jupyter server:
```bash
sh dev_scripts/jupyter_launch_server.sh
```

Launch Jupyter server:
```bash
sh dev_scripts/mlflow_launch_server.sh
```