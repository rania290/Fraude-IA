import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import hashlib
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_connection():
    """Établit une connexion à la base de données."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return connection
    except Error as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        return None

def init_db():
    """Initialise la base de données et crée la table users si elle n'existe pas."""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Créer la table users si elle n'existe pas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            connection.commit()
            logger.info("Base de données initialisée avec succès")
        except Error as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def create_user(username, password, email):
    """Crée un nouvel utilisateur dans la base de données."""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Hash the password before storing
            hashed_password = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                (username, hashed_password, email)
            )
            connection.commit()
            logger.info(f"Utilisateur {username} créé avec succès")
            return True
        except Error as e:
            logger.error(f"Erreur lors de la création de l'utilisateur: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return False

def get_user(username):
    """Récupère les informations d'un utilisateur par son nom d'utilisateur."""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            return user
        except Error as e:
            logger.error(f"Erreur lors de la récupération de l'utilisateur: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return None

def verify_user(username, password):
    """Vérifie les identifiants d'un utilisateur."""
    user = get_user(username)
    if user:
        hashed_password = hash_password(password)
        return user['password'] == hashed_password
    return False 