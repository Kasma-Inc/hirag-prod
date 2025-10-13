from sqlalchemy import TextClause, text

search_by_search_keyword_list: TextClause = text(
    """CREATE OR REPLACE FUNCTION search_by_search_keyword_list (original_token_list character varying[], translation_token_list character varying[], search_keyword_list_original character varying[], search_keyword_list character varying[])
RETURNS TABLE(
    matched_index_list_original integer[],
    matched_index_list_translation integer[]
) AS $$
    from typing import List, Optional
    from rapidfuzz import fuzz

    def find_keyword_matches(
        word_list: List[str],
        search_list: List[str],
        prev_matched_index_list: Optional[List[int]] = None,
    ) -> Optional[List[int]]:
        matched_index_set: Set[int] = set()
        if prev_matched_index_list is not None:
            matched_index_set.update(prev_matched_index_list)
        for j, word in enumerate(word_list):
            for search in search_list:
                if (fuzz.ratio(word, search) > 90) or (
                    (len(word) >= len(search)) and (fuzz.partial_ratio(word, search) > 90)
                ):
                    matched_index_set.add(j)
                    break

        if len(matched_index_set) > 0:
            return sorted(matched_index_set)
        else:
            return None

    matched_index_list_original: Optional[List[int]] = find_keyword_matches(
        original_token_list, search_keyword_list_original
    )
    matched_index_list_translation: Optional[List[int]] = None
    if len(search_keyword_list) > 0:
        if matched_index_list_original is not None:
            matched_index_list_original = find_keyword_matches(
                original_token_list,
                search_keyword_list,
                matched_index_list_original,
            )
        else:
            matched_index_list_translation = find_keyword_matches(
                translation_token_list, search_keyword_list
            )

    yield matched_index_list_original, matched_index_list_translation
$$ LANGUAGE plpython3u;
"""
)

precise_search_by_search_sentence_list: TextClause = text(
    """CREATE OR REPLACE FUNCTION precise_search_by_search_sentence_list (original_text_normalized text, translation_normalized text, search_sentence_list_original character varying[], search_sentence_list character varying[])
RETURNS TABLE(
    fuzzy_match_start_index_list_original INTEGER[],
    fuzzy_match_end_index_list_original INTEGER[],
    fuzzy_match_start_index_list_translation INTEGER[],
    fuzzy_match_end_index_list_translation INTEGER[]
) AS $$
    from typing import List, Dict, Tuple, Optional
    from rapidfuzz import fuzz
    from rapidfuzz.distance import ScoreAlignment

    def find_sentence_matches(
        text_normalized: str, search_list: List[str]
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        fuzzy_match_index_start_list: Optional[List[int]] = []
        fuzzy_match_index_end_list: Optional[List[int]] = []
        queue: List[Tuple[str, int]] = [(text_normalized, 0)]
        while len(queue) > 0:
            text, start_index = queue.pop(0)
            for search in search_list:
                if fuzz.ratio(text, search) > 90:
                    fuzzy_match_index_start_list.append(start_index)
                    fuzzy_match_index_end_list.append(start_index + len(text))
                    break
                elif len(text) >= len(search):
                    match_result: Optional[ScoreAlignment] = fuzz.partial_ratio_alignment(
                        text, search, score_cutoff=90
                    )
                    if match_result is not None:
                        fuzzy_match_index_start_list.append(start_index + match_result.src_start)
                        fuzzy_match_index_end_list.append(start_index + match_result.src_end)
                        if match_result.src_start > 0:
                            queue.append(
                                (
                                    text[: match_result.src_start],
                                    start_index,
                                )
                            )
                        if match_result.src_end < len(text):
                            queue.append(
                                (
                                    text[match_result.src_end :],
                                    start_index + match_result.src_end,
                                )
                            )
        if len(fuzzy_match_index_start_list) > 0:
            return fuzzy_match_index_start_list, fuzzy_match_index_end_list
        else:
            return None, None

    fuzzy_match_start_index_list_original, fuzzy_match_end_index_list_original = find_sentence_matches(
        original_text_normalized, search_sentence_list_original
    )
    fuzzy_match_start_index_list_translation: Optional[List[int]] = None
    fuzzy_match_send_index_list_translation: Optional[List[int]] = None
    if len(search_sentence_list) > 0:
        if fuzzy_match_start_index_list_original is not None:
            search_result_start_index, search_result_end_index = find_sentence_matches(
                original_text_normalized, search_sentence_list
            )
            if search_result_start_index is not None:
                fuzzy_match_start_index_list_original += search_result_start_index
                fuzzy_match_end_index_list_original += search_result_end_index
        else:
            fuzzy_match_start_index_list_translation, fuzzy_match_send_index_list_translation = find_sentence_matches(
                translation_normalized, search_sentence_list
            )

    yield fuzzy_match_start_index_list_original, fuzzy_match_end_index_list_original, fuzzy_match_start_index_list_translation, fuzzy_match_send_index_list_translation
$$ LANGUAGE plpython3u;
"""
)
