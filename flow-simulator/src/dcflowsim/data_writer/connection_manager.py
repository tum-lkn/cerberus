import logging
import peewee

from dcflowsim import constants
from dcflowsim.data_writer import database_model


class DatabaseConnectionManager(object):
    """
    Connection manager for peewee database connections, checks if tables exist
    """

    def __init__(self, database_config):
        """

        Args:
            database_config: dict containing the keys database, user, password, host, port
        """
        self._db_model = database_model
        self._db_config = database_config
        self._db_model.set_connection_parameter(self.db_config)
        self._db_connection = self._db_model.db
        self.logger = logging.getLogger(self.__class__.__name__)
        self.create_all_tables()

    @property
    def interface_type(self):
        return constants.INTERFACE_TYPE_DATABASE

    @property
    def connection(self):
        return self._db_connection

    @property
    def db_config(self):
        return self._db_config

    def disconnect(self):
        self._db_model.db.close()

    def create_all_tables(self):
        self.connection.connect(reuse_if_open=True)
        self.connection.create_tables(self._db_model.TABLES, safe=True)
        self.disconnect()

    def drop_all_tables(self):
        self.connection.connect(reuse_if_open=True)
        self.connection.drop_tables(self._db_model.TABLES, safe=True)
        self.disconnect()

    def clear_all_tables(self):
        self.connection.connect(reuse_if_open=True)
        for table in reversed(self._db_model.TABLES):
            try:
                table.delete().execute()
            except peewee.InternalError:
                self.logger.error("InternalError")
            except peewee.IntegrityError:
                self.logger.error("IntegrityError")
            except peewee.OperationalError:
                self.logger.error("peewee.OperationalError")
        self.disconnect()
