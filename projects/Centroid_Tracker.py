import math

class EuclideanDistTracker:
    def __init__(self):
        # Dicionário para armazenar os IDs
        self.center_points = {}
        self.id_count = 0  # Contador de IDs

    # Método update para atualizar o "deslocamento" dos objetos
    def update(self, objects_rect):
        objects_bbs_ids = []  # Lista para armazenar bbox e IDs

        # Itera sobre "a caixa" do objeto rastreado
        for rect in objects_rect:
            x, y, w, h = rect
            # Calcula o centro da caixa
            center_x = (x + x + w) // 2
            center_y = (y + y + h) // 2
            same_object_detected = False  # Flag para verificar se o objeto foi detectado anteriormente

            # Itera sobre os objetos rastreados e seus centros
            for obj_id, center in self.center_points.items():
                prev_center_x, prev_center_y = center
                # Calcula a distância euclidiana entre os centros dos retângulos
                distance = math.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)

                # Se a distância entre os centros for menor que um limite (25 neste caso), considera-se o mesmo objeto
                if distance < 25:
                    self.center_points[obj_id] = (center_x, center_y)  # Atualiza o centro do objeto rastreado
                    objects_bbs_ids.append([x, y, w, h, obj_id])  # Adiciona retângulo e ID à lista
                    same_object_detected = True  # Marca como objeto detectado
                    break  # Sai do loop, pois o objeto foi encontrado

            # Se não foi detectado o mesmo objeto, adiciona um novo objeto para rastreamento
            if not same_object_detected:
                self.center_points[self.id_count] = (center_x, center_y)  # Adiciona novo centro
                objects_bbs_ids.append([x, y, w, h, self.id_count])  # Adiciona retângulo e novo ID à lista
                self.id_count += 1  # Incrementa o ID para o próximo objeto

        new_center_points = {}
        # Atualiza os centros dos objetos rastreados
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Atualiza o dicionário de centros dos objetos rastreados
        self.center_points = new_center_points.copy()
        return objects_bbs_ids  # Retorna a lista de retângulos e IDs dos objetos detectados
