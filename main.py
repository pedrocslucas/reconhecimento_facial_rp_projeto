from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import mysql.connector
import face_recognition as fr
from io import BytesIO
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)

def conectar_banco():
    return mysql.connector.connect(
        user='root',
        password='123456',
        host='127.0.0.1',
        database='pointjob',
        auth_plugin='caching_sha2_password'
    )

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    try:
        connection = conectar_banco()

        with connection.cursor() as cursor:
            cursor.execute("SELECT nome, foto FROM usuario")
            resultados = cursor.fetchall()

            for nome, foto_blob in resultados:
                foto_rgb = cv2.cvtColor(np.array(Image.open(BytesIO(foto_blob))), cv2.COLOR_BGR2RGB)
                rosto_codificado = fr.face_encodings(foto_rgb)[0]

                rostos_conhecidos.append(rosto_codificado)
                nomes_dos_rostos.append(nome)

    except Exception as e:
        print(f"Erro ao conectar ou obter rostos do banco de dados: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

    return rostos_conhecidos, nomes_dos_rostos

def registrar_presenca(nome_aluno):
    try:
        connection = conectar_banco()

        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO presencas (nome_aluno) VALUES (%s)", (nome_aluno,))
            connection.commit()

        print("Presença registrada com sucesso!")

    except Exception as e:
        print(f"Erro ao conectar ou registrar presença no banco de dados: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

def obter_nomes_presentes():
    nomes_presentes = []

    try:
        connection = conectar_banco()

        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT nome_aluno FROM presencas")
            resultados = cursor.fetchall()

            nomes_presentes = [resultado[0] for resultado in resultados]

    except Exception as e:
        print(f"Erro ao obter nomes dos alunos presentes: {e}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()

    return nomes_presentes

rostos_conhecidos, nomes_dos_rostos = get_rostos()

video_capture = cv2.VideoCapture(0)

@app.route('/fazer_chamada', methods=['POST'])
def fazer_chamada():
    try:
        global rostos_conhecidos, nomes_dos_rostos

        # Processa a imagem recebida do Flutter
        imagem_base64 = request.form['imagem']
        imagem_bytes = base64.b64decode(imagem_base64)
        imagem_np = np.array(Image.open(BytesIO(imagem_bytes)))
        rgb_frame = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2RGB)

        # Corrige a ordem das cores (RGB para BGR)
        rgb_frame = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2BGR)

        localizacao_dos_rostos = fr.face_locations(rgb_frame)
        rosto_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

        pessoas_presentes = set()

        for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos):
            resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)

            face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido)
            melhor_id = np.argmin(face_distances)

            if resultados[melhor_id]:
                nome = nomes_dos_rostos[melhor_id]
                if nome not in pessoas_presentes:
                    registrar_presenca(nome)
                    pessoas_presentes.add(nome)
            else:
                nome = "Desconhecido"

            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(rgb_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Convertendo a imagem de volta para base64 para envio
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        imagem_base64_resultado = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"message": "Chamada realizada com sucesso!", "imagem_resultado": imagem_base64_resultado})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/lista_alunos_presentes', methods=['GET'])
def lista_alunos_presentes():
    try:
        nomes_presentes = obter_nomes_presentes()
        return jsonify({"alunos_presentes": nomes_presentes})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

