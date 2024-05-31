from predict_video import *

Epochs = 30
es = EarlyStopping(monitor='val_binary_accuracy', mode='max', patience=6)

# history = model.fit(train_generator, epochs=Epochs, steps_per_epoch=7, validation_data=validation_generator, validation_steps=25, callbacks=[es])

history = model.fit(train_generator, epochs=Epochs, validation_data=validation_generator, validation_steps=25, callbacks=[es])


# plot model performance

# print(f' history: {history.history.keys()}')

acc = history.history["accuracy"]
print(f"This acc: {acc}")
val_acc = history.history['val_acc']
print(f"This val_acc: {val_acc}")
val_accuracy = history.history['val_acc']
print(f"This val_accuracy: {val_accuracy}")
loss = history.history['loss']
print(f"This loss: {loss}")
val_loss = history.history['val_loss']
print(f"This val_loss: {val_loss}")
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15, 5))
# plt.figure()

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# Validating with the training set
predictions = model.predict(X_train_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]
accuracy = accuracy_score(y_train, predictions)
print('Train Accuracy = %.2f' % accuracy)
confusion_mtx = confusion_matrix(y_train, np.ndarray(predictions).astype(int))
cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)

# Validating with the Validation set
predictions = model.predict(X_val_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]
accuracy = accuracy_score(y_val, predictions)
print('Val Accuracy = %.2f' % accuracy)
confusion_mtx = confusion_matrix(y_val, predictions)
cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)

# Validating on test set
predictions = model.predict(X_test_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]
accuracy = accuracy_score(y_test, predictions)
print('Test Accuracy = %.2f' % accuracy)
confusion_mtx = confusion_matrix(y_test, predictions)
cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)

"""prob_pred = model.predict(X_test_prep)
print(f'prob_ pred: {prob_pred}')"""


ind_list = np.argwhere(False == (y_test == predictions))[:, -1]
if ind_list.size == 0:
	print('There are no miss-classified images.')
else:
	for i in ind_list:
		plt.figure()
		plt.imshow(X_test_crop[i])
		plt.xticks([])
		plt.yticks([])
		plt.title(f'Actual class: {y_val[i]}\nPredicted class: {predictions[i]}')
		plt.show()

ind_list = np.argwhere(False == (y_test == predictions))[:, -1]
if ind_list.size == 0:
	print('There are no miss-classified images.')
else:
	for i in ind_list:
		plt.figure()
		plt.imshow(X_test_crop[i])
		plt.xticks([])
		plt.yticks([])
		plt.title(f'Actual class: {y_val[i]}\nPredicted class: {predictions[i]}')
		plt.show()

print('Accuracy score is :', metrics.accuracy_score(y_test, predictions))
print('Precision score is :', metrics.precision_score(y_test, predictions, average='weighted'))
print('Recall score is :', metrics.recall_score(y_test, predictions, average='weighted'))
print('F1 Score is :', metrics.f1_score(y_test, predictions, average='weighted'))
print('ROC AUC Score is :', metrics.roc_auc_score(y_test, prob_pred, multi_class='ovo', average='weighted'))
print('Cohen Kappa Score:', metrics.cohen_kappa_score(y_test, predictions))
print('\t\tClassification Report:\n', metrics.classification_report(y_test, predictions))
