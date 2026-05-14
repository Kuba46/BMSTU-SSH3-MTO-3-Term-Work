"""
main.py
=======
Полный запуск пайплайна проекта (сбор → очистка → NLP → модели → анализ → визуализации).

Примеры:
	python main.py
	python main.py --with-comments
	python main.py --no-collect --no-train --no-eval
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config.settings import (
	COMMENTS_CLEANED_CSV,
	COMMENTS_PROCESSED_CSV,
	COMMENTS_PREDICTIONS_CSV,
	LABELED_CSV,
	LOGREG_MODEL,
	SVM_MODEL,
	PREDICTIONS_CSV,
	PREDICTIONS_ALL_CSV,
)

log = logging.getLogger(__name__)


def _run_module(module: str, args: list[str] | None = None) -> None:
	cmd = [sys.executable, "-m", module]
	if args:
		cmd.extend(args)
	log.info("Запуск: %s", " ".join(cmd))
	subprocess.run(cmd, check=True)


def _exists(path: Path) -> bool:
	return path.exists()


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s  %(levelname)-8s  %(message)s",
		datefmt="%H:%M:%S",
	)

	parser = argparse.ArgumentParser(
		description="Полный запуск пайплайна проекта (posts + comments)."
	)
	parser.add_argument("--with-comments", action="store_true", help="включить пайплайн комментариев")

	parser.add_argument("--no-collect", action="store_true", help="пропустить сбор данных")
	parser.add_argument("--no-clean", action="store_true", help="пропустить очистку")
	parser.add_argument("--no-dataset", action="store_true", help="пропустить сборку/валидацию корпуса")
	parser.add_argument("--no-comments-merge", action="store_true", help="не объединять комментарии")

	parser.add_argument("--no-nlp", action="store_true", help="пропустить NLP для постов")
	parser.add_argument("--no-comments-nlp", action="store_true", help="пропустить NLP для комментариев")
	parser.add_argument("--no-vectorizer", action="store_true", help="пропустить TF-IDF")
	parser.add_argument("--no-train", action="store_true", help="пропустить обучение моделей")
	parser.add_argument("--no-eval", action="store_true", help="пропустить оценку моделей")
	parser.add_argument("--no-predict", action="store_true", help="пропустить авторазметку")
	parser.add_argument("--no-analysis", action="store_true", help="пропустить анализ")
	parser.add_argument("--no-visuals", action="store_true", help="пропустить визуализации")

	args = parser.parse_args()

	try:
		if not args.no_clean:
			_run_module("data.cleaner")

		if not args.no_dataset:
			_run_module("data.dataset")

		if args.with_comments and not args.no_comments_merge:
			_run_module("data.dataset", ["--comments"])

		if not args.no_nlp:
			_run_module("nlp.preprocessor")
			_run_module("nlp.lemmatizer")

		if args.with_comments and not args.no_comments_nlp:
			if _exists(COMMENTS_CLEANED_CSV):
				_run_module(
					"nlp.preprocessor",
					["--input", str(COMMENTS_CLEANED_CSV), "--output", str(COMMENTS_PROCESSED_CSV)],
				)
				_run_module(
					"nlp.lemmatizer",
					["--input", str(COMMENTS_PROCESSED_CSV), "--output", str(COMMENTS_PROCESSED_CSV)],
				)
			else:
				log.warning("Файл комментариев не найден: %s", COMMENTS_CLEANED_CSV)

		if not args.no_vectorizer:
			_run_module("nlp.vectorizer")

		if not args.no_train:
			if _exists(LABELED_CSV):
				_run_module("models.sentiment")
				_run_module("models.svm_clf")
			else:
				log.warning("Файл разметки не найден: %s", LABELED_CSV)

		if not args.no_eval:
			if _exists(LABELED_CSV):
				_run_module("evaluation.metrics")
			else:
				log.warning("Файл разметки не найден: %s", LABELED_CSV)

		if not args.no_predict:
			if _exists(LOGREG_MODEL) or _exists(SVM_MODEL):
				_run_module("models.predict")
			else:
				log.warning("Модели не найдены: %s, %s", LOGREG_MODEL, SVM_MODEL)

			if args.with_comments and _exists(COMMENTS_PROCESSED_CSV):
				_run_module(
					"models.predict",
					["--input", str(COMMENTS_PROCESSED_CSV), "--output", str(COMMENTS_PREDICTIONS_CSV)],
				)

		if not args.no_analysis:
			_run_module("analysis.aggregator")
			_run_module("analysis.emoji_analyzer")
			_run_module("analysis.event_analysis")

			if args.with_comments and _exists(COMMENTS_PREDICTIONS_CSV):
				_run_module("analysis.aggregator", ["--input", str(COMMENTS_PREDICTIONS_CSV), "--prefix", "comments_"])
				_run_module("analysis.event_analysis", ["--input", str(COMMENTS_PREDICTIONS_CSV), "--prefix", "comments_"])

			if args.with_comments and _exists(PREDICTIONS_CSV) and _exists(COMMENTS_PREDICTIONS_CSV):
				_run_module("analysis.merge_predictions")
				if _exists(PREDICTIONS_ALL_CSV):
					_run_module("analysis.aggregator", ["--input", str(PREDICTIONS_ALL_CSV), "--prefix", "all_"])
					_run_module("analysis.event_analysis", ["--input", str(PREDICTIONS_ALL_CSV), "--prefix", "all_"])

		if not args.no_visuals:
			_run_module("vizualization.plotter_posts")
			_run_module("vizualization.plotter_comments")
			_run_module("vizualization.plotter_emoji")

	except subprocess.CalledProcessError as exc:
		log.error("Шаг завершился с ошибкой (код %s). Остановка.", exc.returncode)
		sys.exit(exc.returncode)


if __name__ == "__main__":
	main()
