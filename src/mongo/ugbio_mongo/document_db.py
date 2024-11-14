import base64
import logging
from datetime import datetime
from json import loads

import boto3
from botocore.exceptions import ClientError
from pymongo import MongoClient, ReturnDocument

logging.getLogger("botocore").setLevel(logging.ERROR)


def retrieve_secrets(secret_name):
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response["SecretBinary"])

    return secret or decoded_binary_secret


class DocumentDBConnect:
    __logger = logging.getLogger(__name__)
    __logger.setLevel(logging.INFO)

    def __init__(self, secret_name):
        config = loads(retrieve_secrets(secret_name))
        connection_uri = config["connection_uri"]
        self.__logger.info("Connecting to document db")
        try:
            self.client = MongoClient(connection_uri)
        except Exception as ex:
            raise Exception(f"could not connect to document db. Exception: {ex}")  # noqa: B904

    def insert_document(self, document, db, collection):
        self.__logger.info(f"Inserting document to collection {collection} in database {db}")
        try:
            result = self.client[db][collection].insert_one(document)
            self.__logger.info(f"Inserted ID {result.inserted_id}")
            return f"{result.inserted_id}"
        except Exception as ex:
            raise Exception(f"Could not insert document to collection {collection} in database {db}. Exception: {ex}")  # noqa: B904

    def remove_immutable_id_field(self, replacement_document):
        if replacement_document.get("_id"):
            self.__logger.debug("Remove immutable field '_id'")
            replacement_document.pop("_id")

    def find_document(self, find_expression, db, collection, projection_expression=None, sort_expression=None):
        self.__logger.info(f"Finding document with {find_expression} in collection {collection} in database {db}")
        try:
            result_cursor = self.client[db][collection].find(
                find_expression, projection=projection_expression, sort=sort_expression
            )
            result = next(result_cursor, None)
            self.__logger.debug(f"Document found:\n{result}")
            return result
        except Exception as ex:
            raise Exception(  # noqa: B904
                f"Could not return document with {find_expression} in collection "
                f"{collection} in database {db}. Exception: {ex}"
            )

    def find_documents(self, find_expression, db, collection, projection_expression=None, sort_expression=None):
        self.__logger.info(
            f"Finding documents with {find_expression} and projection {projection_expression} "
            f"in collection {collection} in database {db}"
        )
        try:
            result_cursor = self.client[db][collection].find(
                find_expression, projection=projection_expression, sort=sort_expression
            )
            result_list = list(result_cursor)
            if not result_list:
                return None
            else:
                return result_list
        except Exception as ex:
            raise Exception(  # noqa: B904
                f"Could not return documents with {find_expression} and projection {projection_expression} "
                f"in collection {collection} in database {db}. Exception: {ex}"
            )

    def find_max(self, field, db, collection):
        self.__logger.info(f"Finding document with maximum {field} in collection {collection} in database {db}")
        try:
            result_cursor = self.client[db][collection].find().sort(field, -1).limit(1)
            result_list = list(result_cursor)
            if not result_list:
                return None
            else:
                return result_list[0]
        except Exception as ex:
            raise Exception(  # noqa: B904
                f"Could not return document with maximum {field} "
                f"in collection {collection} in database {db}. Exception: {ex}"
            )

    def delete_documents(self, filter_expression, db, collection):
        self.__logger.info(
            f"Deleting documents according to filter expression {filter_expression} "
            f"in collection {collection} in database {db}"
        )
        try:
            self.client[db][collection].delete_many(filter_expression)
        except Exception as ex:
            raise Exception(  # noqa: B904
                f"Could not delete documents " f"in collection {collection} in database {db}. Exception: {ex}"
            )

    def find_one_and_replace(  # noqa: PLR0913
        self,
        db,
        collection,
        filter_expression,
        document,
        projection,
        sort=None,
        return_document=ReturnDocument.AFTER,
        array_filters=None,
        hint=None,
        session=None,
        *,
        remove_id=False,
        upsert=True,
    ):
        self.__logger.info(
            f"Updating document according to filter expression {filter_expression} "
            f"in collection {collection} in database {db}"
        )
        if remove_id:
            self.remove_immutable_id_field(document)
        try:
            result = self.client[db][collection].find_one_and_replace(
                filter_expression,
                document,
                upsert=upsert,
                projection=projection,
                sort=sort,
                return_document=return_document,
                array_filters=array_filters,
                hint=hint,
                session=session,
            )

            return f"{result.get('_id')}"
        except Exception as ex:
            raise Exception(  # noqa: B904
                f"Could not replace document in collection {collection} "
                f"in database {db} with filter {filter_expression}. Exception: {ex}"
            )

    def find_one_and_update(  # noqa: PLR0913
        self,
        db,
        collection,
        filter_expression,
        update,
        projection,
        sort=None,
        return_document=ReturnDocument.AFTER,
        array_filters=None,
        hint=None,
        session=None,
        *,
        upsert=True,
    ):
        self.__logger.info(
            f"Updating document according to filter expression {filter_expression} "
            f"in collection {collection} in database {db}"
        )
        try:
            result = self.client[db][collection].find_one_and_update(
                filter_expression,
                update,
                upsert=upsert,
                projection=projection,
                sort=sort,
                return_document=return_document,
                array_filters=array_filters,
                hint=hint,
                session=session,
            )
        except Exception as ex:
            raise Exception(  # noqa: B904
                f"Could not update document in collection {collection} "
                f"in database {db} with filter {filter_expression}. Exception: {ex}"
            )
        else:
            return result

    def close_connection(self):
        self.__logger.info("Closing connection to document db")
        try:
            self.client.close()
        except Exception:
            self.__logger.error("Could not close connection to document db. Exception: {ex}")

    def preprocess_key(self, json_key):
        """Replace characters invalid in MongoDB."""
        result = json_key
        result = result.replace("$", "__")
        result = result.replace(".", "__")
        return result

    def preprocess_keys_shallow(self, json_dict):
        """Replace characters which are invalid in keys at first level."""
        result = {}
        for k, _ in json_dict.items():
            k_new = self.preprocess_key(k)
            result[k_new] = json_dict[k]
        return result

    def convert_datetime_from_isoformat(self, datetime_isoformat):
        return datetime.fromisoformat(datetime_isoformat)
