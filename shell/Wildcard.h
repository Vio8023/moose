/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _WILDCARD_H
#define _WILDCARD_H

// Just a couple of extern definitions for general use.

/**
 * simpleWildcardFind returns the number of Ids found.
 * This is the basic wildcardFind function, working on a single
 * tree. It adds entries into the vector 'ret' with Ids found according
 * to the path string. It preserves the order of the returned Ids
 * as the order of elements traversed in the search. It does NOT
 * eliminate duplicates. This is a depth-first search.
 * Note that it does NOT list every entry in arrays, only the base Element.
 *
 * Wildcard search rules are like this:
 * - Multiple wildcard paths can be specified, separated by commas
 * - Each wildcard path looks like a file path, separated by '/'
 * - At each level of the wildcard path, we have element specifiers.
 * - Each specifier can be a single fully specified element, or a wildcard.
 * - Substrings of the specifier can be wildcarded with '#'
 *   - The entire specifier can also be a '#'
 * - Characters of the specifier can be wildcarded with '?'
 * - An entire recursive tree can be specified using '##'
 * - As a qualifier to any of the wildcard specifiers we can use expressions
 *   enclosed in square brackets. These expressions are of the form:
 *   [TYPE==<string>]
 *   [TYPE=<string>]
 *   [CLASS=<string>]
 *   [ISA=<string>]
 *   [ISA==<string>]
 *   [TYPE!=<string>]
 *   [CLASS!=<string>]
 *   [ISA!=<string>]
 *   [FIELD(<fieldName)=<string>]
 */
int simpleWildcardFind( const string& path, vector<ObjId>& ret);


/**
 * wildcardFind returns the number of Ids found.
 * This behaves the same as simpleWildcardFind, except that it eliminates
 * non-unique entries, and in the process will scramble the ordering.
 */
int wildcardFind(const string& n, vector<ObjId>& ret);

/**
 * Recursive function to compare all descendants and cram matches into ret.
 * Returns number of matches.
 */
int allChildren( ObjId start, const string& insideBrace, vector< ObjId >& ret );

#endif // _WILDCARD_H
