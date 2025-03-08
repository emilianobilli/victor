# database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import numpy as np

# 📌 Definir la base ORM
Base = declarative_base()

# 📌 Configurar la conexión a SQLite
DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL, echo=False)  # Cambia `echo=True` a `False` para evitar logs excesivos
SessionLocal = sessionmaker(bind=engine)

# 📍 Tabla de Usuarios
class Data(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vector_id = Column(Integer, unique=True, nullable=False, index=True)
    rawdata = Column(String, nullable=False)  # JSON almacenado como string

    def to_dict(self):
        return {
            "id": self.id,
            "vector_id": self.vector_id,
            "metadata": json.loads(self.rawdata),
        }

# 📍 Tabla de Vectores
class Vector(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding = Column(LargeBinary, nullable=False)  # Almacena los vectores en binario

    def get_vector(self):
        return np.frombuffer(self.embedding, dtype=np.float32)

# 📌 Función para inicializar la base de datos
def init_db():
    """Crea las tablas en la base de datos si no existen."""
    Base.metadata.create_all(engine)

# 📌 Función para obtener una sesión de la base de datos
def get_session():
    """Retorna una nueva sesión de la base de datos."""
    return SessionLocal()
