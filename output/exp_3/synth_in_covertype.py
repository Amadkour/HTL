# evaluate a weighted average ensemble for classification
import time
import json
from sklearn.metrics import accuracy_score

from river.drift import DDM, ADWIN
from sklearn.model_selection import train_test_split
from strlearn.metrics import geometric_mean_score_1, precision, recall, balanced_accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
# get a list of base models
import helper

K_MAX = 10
num_iterations = 1
metrics = [balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, ]
flows = [
    'CORAL',
    'CDTL',
    'HTL'
]
statistics = [{'time': 0, 'number_of_updates': 0} for _ in flows]

# streams = helper.synthetic_streams_10(123)
streams = helper.realstreams2()

scores = np.zeros((len(flows), streams[list(streams.keys())[0]].n_chunks - 1, len(metrics)))
sources_domain = [
    streams[list(streams.keys())[0]].get_chunk(),
                  streams[list(streams.keys())[0]].get_chunk(),
                  streams[list(streams.keys())[0]].get_chunk(),
                  # streams[list(streams.keys())[0]].get_chunk(),
                  # streams[list(streams.keys())[0]].get_chunk(),
                  # streams[list(streams.keys())[0]].get_chunk()
                  ]
source_weights = [1 for _ in range(len(sources_domain))]
weights=[]

def stream_flow(flow_index, flow):
    global statistics, sources_domain,source_weights,weights
    weights=[]
    streams = helper.realstreams()
    models = []
    source_weights = [1 for _ in range(len(sources_domain))]

    index = 0
    start_time = time.perf_counter()
    while True:
        print('====[ chunk-', index, ' ]====')
        if streams['covertype'].is_dry():
            break
        chunk = streams['covertype'].get_chunk()
        X, y = chunk
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
        if index == 0:
            evaluate_models(X_train, y_train, models)

            # update sources dim
            sources_domain = homogenous(X,sources_domain)

            if flow != flows[2]:
                weights = calc_weights(X_train, y_train, models)
            else:
                weights = calc_proposed_weights(X_train, y_train, models, flow)
        else:
            evaluate_models(X_train, y_train, models)
            if flow != flows[2]:
                weights = calc_weights(X_train, y_train, models)
            else:
                weights = calc_proposed_weights(X_train, y_train, models, flow)
        if flow == flows[0]:
            source_classifiers = calc_source_classifiers_with_coral(sources_domain, X, y, models)
        else:
            source_classifiers = calc_source_classifiers_with_cdtl(sources_domain, X, y, models)

        old_statistics = statistics[flow_index]
        statistics[flow_index]['number_of_updates'] = old_statistics['number_of_updates'] + 1

        yhat = [predict(X_test[i], models, source_classifiers, weights, source_weights) for i in
                range(len(X_test))]
        calculate_scores(flow_index, index - 1, y_test, yhat)
        # print('========================[ ', flow, ':', scores[flow_index, index - 1], ']')

        index += 1
    finish_time = time.perf_counter()
    statistics[flow_index]['time'] = (finish_time - start_time)
    np.save(f"output/sub_score_" + flow, scores)
    file_path = f"output/sub_statistics_" + flow + ".text"
    # Save the list of dictionaries to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(statistics, json_file)


# evaluate each base model


def homogenous(X, sources):
    # update sources dim
    new_sources_domain = []
    for (x, y) in sources:
        if np.shape(x)[1] > np.shape(X)[1]:
            # Calculate eigenvalues and eigenvectors
            # Determine the desired size for the square matrix (assuming larger dimension)
            desired_size = max(x.shape)

            # Create a new square matrix filled with zeros
            square_matrix = np.zeros((desired_size, desired_size))

            # Copy the values from the non-square matrix into the upper-left corner of the square matrix
            square_matrix[:X.shape[0], :X.shape[1]] = X

            eigenvalues, eigenvectors = np.linalg.eig(square_matrix)

            # Extract the eigenvector corresponding to the maximum eigenvalue
            best_eigenvector = eigenvectors[:, :np.shape(X)[1]]
            # Normalize the eigenvector if needed
            best_eigenvector = best_eigenvector / np.linalg.norm(best_eigenvector)
        elif np.shape(x)[1] < np.shape(X)[1]:
            # Calculate eigenvalues and eigenvectors

            desired_size = max(X.shape)

            # Create a new square matrix filled with zeros
            square_matrix = np.zeros((desired_size, desired_size))

            # Copy the values from the non-square matrix into the upper-left corner of the square matrix
            square_matrix[:x.shape[0], :x.shape[1]] = x

            _, eigenvectors = np.linalg.eig(square_matrix)
            # Extract the eigenvector corresponding to the maximum eigenvalue
            best_eigenvector = eigenvectors[:, :np.shape(X)[1]]

            # Normalize the eigenvector if needed
            best_eigenvector = best_eigenvector / np.linalg.norm(best_eigenvector)
        else:
            best_eigenvector = x

        new_x = np.real(best_eigenvector)
        new_y = y[:len(best_eigenvector)]
        new_sources_domain.append((new_x, new_y))

    return new_sources_domain.copy()


def evaluate_models(X_train, y_train, models):
    # r = RandomForestClassifier(n_estimators=100)

    if len(models) > K_MAX - 1:
        ac = [np.min(c) for c in weights]
        worst_index = ac.index(np.min(ac))
        models.pop(worst_index)
        weights.pop(worst_index)
    g = GradientBoostingClassifier(n_estimators=100)
    g.fit(X_train, y_train)
    models.append((np.unique(y_train), g))


def calc_weights(x, y, models):
    global weights
    classes = np.unique(y)
    weights = [[1.0 for _ in range(len(classes))] for _ in range(len(models))]

    for classifier_index, (_, k) in enumerate(models):
        for class_index, c in enumerate(classes):
            predictions = k.predict(x)
            weight = len(predictions[predictions == c]) / len(y[y == c])
            weights[classifier_index][class_index] = weight
    return weights


def calc_proposed_weights(x, y, models, flow):
    classes = np.unique(y)
    weights = [[1.0 for _ in range(len(classes))] for _ in range(len(models))]
    for classifier_index, (_, k) in enumerate(models):
        for class_index, c in enumerate(classes):
            predictions = k.predict(x)
            Wi = (len(predictions[predictions == c]) / len(y[y == c])) * (
                    len(predictions[predictions != c]) / len(y[y != c]))
            weights[classifier_index][class_index] = Wi
    return weights


def ensemble_support_matrix(X, pool_classifiers):
    """Ensemble support matrix."""
    return np.array([member_clf.predict_proba([X]) for member_clf in pool_classifiers])


def predict(data, pool_target_classifiers, pool_source_classifiers, pool_target_weights, pool_source_weights):
    target_clf = [model for y, model in pool_target_classifiers if len(y) == len(pool_target_classifiers[-1][0])]
    source_clf = [model for y, model in pool_source_classifiers if len(y) == len(pool_target_classifiers[-1][0])]
    if len(source_clf) == 0:
        esm_target = ensemble_support_matrix(data, target_clf)
        esm_target = np.array(esm_target).reshape(len(esm_target), len(pool_target_weights[0]))
        Ft = [[0.0 for _ in range(len(pool_target_weights[0]))] for _ in range(len(pool_target_weights))]
        for i in range(len(esm_target)):
            for j in range(len(pool_target_weights[i])):
                Ft[i][j] = esm_target[i][j] * pool_target_weights[i][j]
        target_matrix = np.mean(Ft, axis=0)
        return np.argmax(target_matrix, axis=0)

    else:
        esm_target = ensemble_support_matrix(data, target_clf)
        esm_source = ensemble_support_matrix(data, source_clf)
        esm_target = np.array(esm_target).reshape(len(esm_target), len(pool_target_weights[0]))
        esm_source = np.array(esm_source).reshape(len(esm_source), len(esm_source[0][0]))
        Ft = [[0.0 for _ in range(len(esm_target[0]))] for _ in range(len(esm_target))]

        '''target prediction'''
        '''use esm length instead of weight length because of using subset of classifiers equal to esm length'''
        for i in range(len(esm_target)):
            for j in range(len(esm_target[i])):
                Ft[i][j] = esm_target[i][j] * pool_target_weights[i][j]
        target_matrix = np.argmax(Ft, axis=1)

        # pred = np.argmax(target_matrix, axis=0)
        '''source prediction'''
        F_est = [[(x * pool_source_weights[i]) for j, x in enumerate(esm_source[i])] for i in range(len(esm_source))]
        F_est = np.mean(F_est, axis=0)
        Ft = np.mean(Ft, axis=0)
        F_est = [F_est[i] + Ft[i] for i in range(len(Ft))]
        return np.argmax(F_est, axis=0)


def aw_colar(DS, wS, DT):
    # Unpack source domain
    from scipy import linalg

    XS, yS = DS
    # Calculate covariance matrices
    CS = (np.cov(XS, rowvar=False) +
          np.eye(XS.shape[1]))
    CT = np.cov(DT[0], rowvar=False) + np.eye(DT[0].shape[1])

    # Apply transformations
    multiplier = linalg.fractional_matrix_power(CS, -0.5)
    multiplier[np.isnan(multiplier)] = 0
    multiplier[np.isinf(multiplier)] = 0
    XS = wS * XS.dot(multiplier)
    multiplier = linalg.fractional_matrix_power(CT, 0.5)
    multiplier[np.isnan(multiplier)] = 0
    multiplier[np.isinf(multiplier)] = 0
    XS = XS.dot(multiplier)

    # Pack the transformed source domain
    DS_p = (np.real(XS), yS)
    return DS_p

    # DS_p now contains the transformed source domain


def projected_data(chunk, sources, source_instance_weights):
    projected_sources = []
    for index, source in enumerate(sources):
        projected_source = aw_colar(source, source_instance_weights[index], chunk)
        projected_sources.append(projected_source)
    return projected_sources


def calc_source_classifiers_with_coral(data_sources, Tx, Ty, models):
    classifiers = []
    for ite in range(num_iterations):
        for (Sx, Sy) in projected_data((Tx, Ty), data_sources, source_weights):
            classifier = GradientBoostingClassifier(n_estimators=100)
            classifier.fit(Sx, Sy)
            classifiers.append((np.unique(Sy), classifier))
    return classifiers


def calc_source_classifiers_with_cdtl(data_sources, Tx, Ty, models):
    classifiers = []
    for ite in range(num_iterations):
        for source_index, (Sx, Sy) in enumerate(projected_data((Tx, Ty), data_sources, source_weights)):
            classifier = GradientBoostingClassifier(n_estimators=100)
            classifier.fit(Sx, Sy)

            yhat = classifier.predict(Sx)
            acc = balanced_accuracy_score(Ty, yhat)
            source_weights[source_index] = acc
            classifiers.append((np.unique(Sy), classifier))

        for index, (x, y) in enumerate(data_sources):
            yhat = [predict(x[i], models, classifiers, weights, source_weights) for i in range(len(x))]

            accuracy = accuracy_score(y, yhat)
            # Calculate beta for weight multiplication
            beta = 0.5 * np.log(1 + np.sqrt(2 * np.log(x.shape[0] / num_iterations)))
            source_weights[index] = (accuracy * np.exp(-beta * len(y[y == yhat])))
    return classifiers


def calculate_scores(flow_index, chunk_index, y, y_pred):
    scores[flow_index, chunk_index] = [metric(np.array(y), np.array(y_pred)) for metric in metrics]


if __name__ == '__main__':
    start_time = time.perf_counter()

    for flow_index, flow in enumerate(flows):
        stream_flow(flow_index, flow)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")

    np.save(f"output/score", scores)
    file_path = f"output/statistics.text"
    # Save the list of dictionaries to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(statistics, json_file)

    # from joblib import Parallel, delayed
    #
    # start_time = time.perf_counter()
    # result = Parallel(n_jobs=4, prefer="threads")(
    #     delayed(stream_flow)(flow_index,flow) for flow_index, flow in enumerate(flows))
    # finish_time = time.perf_counter()
    # print(f"Program finished in {finish_time - start_time} seconds")
    # print(result)
