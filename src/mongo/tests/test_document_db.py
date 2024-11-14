import unittest
from unittest.mock import patch

from ugbio_mongo.document_db import DocumentDBConnect

DB = "test_db"
COLLECTION = "test_collection"


class TestDocumentDBConnect(unittest.TestCase):
    @patch("ugbio_mongo.document_db.retrieve_secrets")
    @patch("ugbio_mongo.document_db.MongoClient")
    def setUp(self, mock_mongo_client, mock_retrieve_secrets):
        # Mock the retrieval of secrets
        mock_retrieve_secrets.return_value = '{"connection_uri": "mock_uri"}'

        # Instantiate the DocumentDBConnect class, which should use the mock MongoClient
        self.db_connect = DocumentDBConnect(secret_name="test_secret")

        # Set up the mock MongoClient and mock database
        self.mock_client = mock_mongo_client.return_value
        self.mock_db = self.mock_client[DB]
        self.mock_collection = self.mock_db[COLLECTION]

    def test_insert_document(self):
        document = {"_id": 1, "name": "test"}
        self.mock_collection.insert_one.return_value.inserted_id = "mock_id"

        result = self.db_connect.insert_document(document, DB, COLLECTION)

        self.assertEqual(result, "mock_id")
        self.mock_collection.insert_one.assert_called_once_with(document)

    def test_find_document(self):
        find_expression = {"name": "test"}
        self.mock_collection.find.return_value = iter([{"_id": 1, "name": "test"}])

        result = self.db_connect.find_document(find_expression, DB, COLLECTION)

        self.assertEqual(result, {"_id": 1, "name": "test"})
        self.mock_collection.find.assert_called_once_with(find_expression, projection=None, sort=None)

    def test_find_documents(self):
        find_expression = {"name": "test"}
        self.mock_collection.find.return_value = iter([{"_id": 1, "name": "test"}, {"_id": 2, "name": "example"}])

        results = self.db_connect.find_documents(find_expression, DB, COLLECTION)

        self.assertEqual(results, [{"_id": 1, "name": "test"}, {"_id": 2, "name": "example"}])

    def test_find_max(self):
        self.mock_collection.find.return_value.sort.return_value.limit.return_value = iter(
            [{"_id": 1, "max_field": 100}]
        )

        result = self.db_connect.find_max("max_field", DB, COLLECTION)

        self.assertEqual(result, {"_id": 1, "max_field": 100})

    def test_delete_documents(self):
        filter_expression = {"name": "test"}

        self.db_connect.delete_documents(filter_expression, DB, COLLECTION)

        self.mock_collection.delete_many.assert_called_once_with(filter_expression)

    def test_find_one_and_replace(self):
        filter_expression = {"_id": 1}
        document = {"name": "updated_test"}
        self.mock_collection.find_one_and_replace.return_value = {"_id": 1, "name": "updated_test"}

        result = self.db_connect.find_one_and_replace(
            DB, COLLECTION, filter_expression, document, projection=None, upsert=True
        )

        self.assertEqual(result, "1")
        self.mock_collection.find_one_and_replace.assert_called_once_with(
            filter_expression,
            document,
            projection=None,
            sort=None,
            return_document=True,
            array_filters=None,
            hint=None,
            session=None,
            upsert=True,
        )

        filter_expression = {"_id": 1}
        update = {"$set": {"name": "new_test"}}
        self.mock_collection.find_one_and_update.return_value = {"_id": 1, "name": "new_test"}

        result = self.db_connect.find_one_and_update(
            DB, COLLECTION, filter_expression, update, projection=None, upsert=True
        )

        self.assertEqual(result, {"_id": 1, "name": "new_test"})
        self.mock_collection.find_one_and_update.assert_called_once_with(
            filter_expression,
            update,
            projection=None,
            sort=None,
            return_document=True,
            array_filters=None,
            hint=None,
            session=None,
            upsert=True,
        )


if __name__ == "__main__":
    unittest.main()
